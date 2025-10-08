import neon
import warp as wp
import numpy as np
import os, sys, time, trimesh
import matplotlib.pyplot as plt

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
    DoNothingBC,
    ZouHeBC,
    HybridBC,
)
from xlb.operator.boundary_masker import MeshVoxelizationMethod
from xlb.utils.mesher import make_cuboid_mesh, MultiresIO
from xlb.utils.makemesh import generate_mesh
from xlb.operator.force import MultiresMomentumTransfer
from xlb.helper.initializers import CustomMultiresInitializer
from xlb import MresPerfOptimizationType
import httpx, logging, getopt, json
from json.decoder import JSONDecodeError
from uuid import uuid4
from threading import Thread
from typing import Any
# Use 8 CPU devices if running on ACP
acp_env = os.environ.get('ACP_ENVIRONMENT', '')
if acp_env not in ('', 'local'):
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

WORKER_PROTOCOL = os.environ.get('SCM_PROTOCOL', '')
WORKER_HOST = os.environ.get('SCM_HOST', '')
WORKER_PORT = os.environ.get('SCM_PORT', '')

HEARTBEAT_SLEEP = int(float(os.environ.get('SCM_SOLVERHEARTBEAT', 1000)) / 1000)
HEARTBEAT_THREAD = None
HEARTBEAT_CANCELLED = False

### SCM Functions ###
def running_via_scm():
    """
    Checks if the code is running via the SCM worker protocol.

    Returns:
        bool: True if WORKER_PROTOCOL is set (indicating execution via SCM), False otherwise.
    """

    if WORKER_PROTOCOL:
        return True

    return False

def scm_event(endpoint, data=0, event_id=''):
    """
    Sends an event to a specified SCM worker endpoint using HTTP POST and returns the response.

    Args:
        endpoint (str): The endpoint path to send the event to.
        data (int, optional): The data payload to send. Defaults to 0.
        event_id (str, optional): An identifier for the event. Defaults to ''.

    Returns:
        Any: The 'response' field from the JSON response if available, otherwise the provided event_id.

    Notes:
        - If any of WORKER_PROTOCOL, WORKER_HOST, WORKER_PORT, or endpoint are not set, returns the event_id.
        - If the response cannot be decoded as JSON, returns the event_id.
    """

    if not endpoint or not WORKER_PROTOCOL or not WORKER_HOST or not WORKER_PORT:
        return event_id

    url = f'{WORKER_PROTOCOL}://{WORKER_HOST}:{WORKER_PORT}{endpoint}'

    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        'data': data,
        'id': event_id,
    }

    response = httpx.post(url, headers=headers, json=data)

    try:
        return response.json().get('response', event_id)
    except JSONDecodeError:
        return event_id

    return event_id

def heartbeat():
    """
    Continuously sends a heartbeat signal to the compute worker endpoint to indicate the process is alive.

    The function repeatedly calls the `scm_event` function with the '/ComputeWorker/v1/heartbeat' endpoint.
    If the response is 'canceled' or the global variable `HEARTBEAT_CANCELLED` is set to True, the loop breaks and the function returns.
    Otherwise, the function sleeps for a duration specified by the global variable `HEARTBEAT_SLEEP` before sending the next heartbeat.

    Returns:
        None
    """

    while True:
        response = scm_event('/ComputeWorker/v1/heartbeat')

        if response == 'canceled' or HEARTBEAT_CANCELLED:
            return

        time.sleep(HEARTBEAT_SLEEP)

def scm_init():
    """
    Performs SCM initialization by attaching to the compute worker and starting the heartbeat thread.

    This function performs the following actions:
    1. Sends an attach event to the compute worker endpoint.
    2. Creates and starts a global heartbeat thread to maintain regular communication and status checks.

    Globals:
        HEARTBEAT_THREAD: Thread object responsible for running the heartbeat function.

    Side Effects:
        Modifies the global HEARTBEAT_THREAD variable and starts a new thread.
    """

    global HEARTBEAT_THREAD

    scm_event('/ComputeWorker/v1/attach', 1)

    HEARTBEAT_THREAD = Thread(target=heartbeat)
    HEARTBEAT_THREAD.start()

    scm_progress(0)

def scm_progress(progress):
    """
    Sends a progress update to the SCM compute worker.

    Args:
        progress (int): The progress value to send, between 0 and 100.

    Returns:
        None
    """

    scm_event('/ComputeWorker/v1/progress', progress)

def scm_results_available(final_update=False):
    """
    Notifies that results are available by sending an event to the '/ComputeWorker/v1/results' endpoint.

    Args:
        final_update (bool, optional): Indicates whether this is the final update. Defaults to False.

    Returns:
        None
    """

    scm_event('/ComputeWorker/v1/results', int(final_update))

def scm_cancel_heartbeat():
    """
    Cancels the ongoing heartbeat process by setting the HEARTBEAT_CANCELLED flag to True.
    If a heartbeat thread is running, waits for it to finish and then resets the thread reference.
    """

    global HEARTBEAT_CANCELLED
    global HEARTBEAT_THREAD

    HEARTBEAT_CANCELLED = True
    if HEARTBEAT_THREAD:
        HEARTBEAT_THREAD.join()
        HEARTBEAT_THREAD = None

def scm_set_error(code, message):
    """
    Sets an error state by sending an error code and message to the ComputeWorker event handler.

    Args:
        code (int): The error code representing the type of error.
        message (str): A descriptive message explaining the error.

    Returns:
        None

    Side Effects:
        Triggers the '/ComputeWorker/v1/seterror' event with the provided code and message.
    """

    scm_event('/ComputeWorker/v1/seterror', code, message)

def scm_complete():
    """
    Notifies the SCM worker that the process is complete by sending a completion event.

    Returns:
        None
    """

    scm_progress(100)

    scm_event('/ComputeWorker/v1/complete', 1, str(uuid4()))

    scm_cancel_heartbeat()
####################

wp.clear_kernel_cache()
wp.config.quiet = False

def prep_inputs(input_file):
    start_time = time.time()
    f = open(input_file)
    jsonfile = json.load(f)
    proj_path = os.path.dirname(os.path.abspath(input_file))
    jsonfile['projPath'] = proj_path
    settings = jsonfile['settings']
    voxel_size = settings['voxelSize']
    ulb = settings['ulb']
    # Extract the inlet velocity from the json dict
    prescribed_velocity_phys = jsonfile['InletBC']['x']
    if running_via_scm():
        output_dir = proj_path
    else:
        output_dir = os.path.join(proj_path, jsonfile['outputName'])    
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for fx in [os.path.join(output_dir,f) for f in os.listdir(output_dir)]:
            os.remove(fx)
    
        
    with open(os.path.join(output_dir, "project.log"),'w') as fd:
        fd.write("***  Studio Wind Tunnel Solver Log File ***\n\n\n")
        fd.write("Date Created: "+time.asctime(time.localtime())+" \n\n")  
        fd.write("Processing input json ... \n\n") 
    logging.info("Processing input json ...")

    # Set accuracy and lattice type
    if settings['doublePrecision']==True:
        precision_policy = PrecisionPolicy.FP64FP64
    elif settings['doublePrecision']==-1:
        precision_policy = PrecisionPolicy.FP16FP16
    else:
        precision_policy = PrecisionPolicy.FP32FP32
    
    compute_backend = ComputeBackend.NEON
    velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
  
    ### Process Car for obj and scale
    body_stl = os.path.join(proj_path, str(jsonfile['vehicle']['body']))
    filename, file_extension = os.path.splitext(body_stl)    
        
    body_mesh = trimesh.load_mesh(body_stl, process=False)        
    if  file_extension =='.obj':
        body_mesh.apply_scale(0.01)
        body_mesh.export(os.path.join(output_dir, filename+'.stl'))
        body_mesh = trimesh.load_mesh(os.path.join(output_dir, filename+'.stl'), process=False)        
    
    #If any wheels listed
    if len(jsonfile['vehicle']['wheels']) > 0:
        wheel_stls = []
        for wheel in jsonfile['vehicle']['wheels']:
            wheel = os.path.join(proj_path, wheel)
            wheel_stls.append(wheel)    
        wheel_meshes =[]
        w=1
        for wheel in wheel_stls:
            wheel_mesh = trimesh.load_mesh(wheel, process=False)
            if file_extension =='.obj':
                wheel_mesh.apply_scale(0.01)        
                wheel_mesh.export(os.path.join(output_dir, 'wheel'+str(w)+'.stl'))
                wheel_mesh = trimesh.load_mesh(os.path.join(output_dir, 'wheel'+str(w)+'.stl'))
            w+=1                
            wheel_meshes.append(wheel_mesh) 
            
        car_mesh = trimesh.util.concatenate(body_mesh + wheel_meshes)
    else:
        car_mesh = body_mesh
        wheel_meshes=None
 
    # ===========
    # Initialize XLB
    xlb.init(
        velocity_set=velocity_set,
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )

    
    level_data, body_vertices, wheel_vertices, grid_shape_zip, partSize, actual_num_levels, shift, sparsity_pattern, level_origins = mesh_prep(
            voxel_size, car_mesh, body_mesh, wheel_meshes, output_dir, jsonfile
        )

    
    # Characteristic length
    L = float(partSize[0])    
    #Material Setup
    material = jsonfile['fluid']
    density = material['density']
    dynamic_viscosity = material['viscosity']
    kinematic_viscosity = dynamic_viscosity / density

    # Compute Re   
    Re = abs(prescribed_velocity_phys) * L / kinematic_viscosity

    # Calculate lattice parameters
    delta_x_coarse = voxel_size * 2 ** (actual_num_levels - 1)
    delta_t = voxel_size * ulb / prescribed_velocity_phys
    lbm_visc = kinematic_viscosity * delta_t / (voxel_size ** 2)
    omega = 1.0 / (3.0 * lbm_visc + 0.5)

    # Define exporter objects

    field_name_cardinality_dict = {"velocity": 3, "density": 1}
    h5exporter = MultiresIO(
        field_name_cardinality_dict,
        level_data,
        scale=voxel_size,
        offset=-shift,
        timestep_size=delta_t,
    )
    bc_mask_exporter = MultiresIO({"bc_mask": 1}, level_data, scale=voxel_size, offset=-shift)

    # Create grid
    grid = multires_grid_factory(
        grid_shape_zip,
        velocity_set=velocity_set,
        sparsity_pattern_list=sparsity_pattern,
        sparsity_pattern_origins=[neon.Index_3d(*box_origin) for box_origin in level_origins],
    )
    # Calculate num_steps
    coarsest_level = grid.count_levels - 1
    grid_shape_x_coarsest = grid.level_to_shape(coarsest_level)[0]
    if jsonfile['settings']['flowPasses'] > 0:
        num_steps = int(jsonfile['settings']['flowPasses'] * (grid_shape_x_coarsest / ulb))
    else:
        num_steps = int(jsonfile['settings']['iterations'])


    # Setup boundary conditions
    boundary_conditions = setup_boundary_conditions(grid, level_data, body_vertices, wheel_vertices, ulb, lbm_visc, grid_shape_zip, precision_policy, jsonfile, velocity_set, compute_backend)

    # Create initializer
    initializer = CustomMultiresInitializer(
    bc_id=boundary_conditions[-2].id,  # bc_outlet
    constant_velocity_vector=(ulb, 0.0, 0.0),
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=compute_backend,
    )

    # Initialize simulation   
    sim = xlb.helper.MultiresSimulationManager(
        omega=omega,
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="KBC",
        initializer=initializer,
        mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST,
    )
    
    # Compute voxel statistics and reference area
    stats = compute_voxel_statistics_and_reference_area(sim, bc_mask_exporter, level_data, actual_num_levels, sparsity_pattern, boundary_conditions, voxel_size)
    active_voxels = stats["active_voxels"]
    solid_voxels = stats["solid_voxels"]
    total_voxels = stats["total_voxels"]
    total_lattice_updates_per_step = stats["total_lattice_updates_per_step"]
    reference_area = stats["reference_area"]
    reference_area_physical = stats["reference_area_physical"]
             
    wp.synchronize()

    # Setup momentum transfer    
    momentum_transfer = MultiresMomentumTransfer(
        boundary_conditions[-1],
        mres_perf_opt=xlb.MresPerfOptimizationType.FUSION_AT_FINEST,
        compute_backend=compute_backend,
    )

    with open(os.path.join(output_dir, "project.log"),'a') as fd:
        fd.write('Material Properties\n')
        fd.write('___________________\n')
        fd.write(f'Density:  {density:.4f} kg/m3\n')
        fd.write(f'Visc Dyn: {dynamic_viscosity:.4e} Pa-s\n')
        fd.write(f'Visc Kin: {kinematic_viscosity:.4e} m2/s\n')
        fd.write(f'Visc LBM: {lbm_visc:.4e} \n\n')
        fd.write('Boundary Setup\n')
        fd.write('___________________\n')
        fd.write(f"Walls: {jsonfile['BCtypes']['walls']}\n")
        fd.write(f"Ground: {jsonfile['BCtypes']['ground']}\n")
        fd.write('\nSolver Parameters\n')
        fd.write('___________________\n')
        fd.write(f"Number of flow passes: {jsonfile['settings']['flowPasses']}\n")
        fd.write(f"Calculated iterations: {num_steps:,}\n")
        fd.write(f"Finest voxel size: {voxel_size} meters\n")
        fd.write(f"Coarsest voxel size: {delta_x_coarse} meters\n")
        fd.write(f"Total voxels: {sum(np.count_nonzero(mask) for mask in sparsity_pattern):,}\n")
        fd.write(f"Total active voxels: {total_voxels:,}\n")
        fd.write(f"Active voxels per level: {active_voxels}\n")
        fd.write(f"Solid voxels per level: {solid_voxels}\n")
        fd.write(f"Total lattice updates per global step: {total_lattice_updates_per_step:,}\n")
        fd.write(f"Actual number of refinement levels: {actual_num_levels}\n")
        fd.write(f"Physical inlet velocity: {prescribed_velocity_phys:.4f} m/s\n")
        fd.write(f"Lattice velocity (ulb): {ulb}\n")
        fd.write(f"Characteristic length: {L: .4f} meters\n")
        fd.write(f"Computed reference area (bc_mask): {reference_area} lattice units\n")
        fd.write(f"Physical reference area (bc_mask): {reference_area_physical:.6f} m^2\n")
        fd.write(f"Reynolds number: {Re:,.2f}\n")
        fd.write(f'Inlet Velocity:    {prescribed_velocity_phys:.1f} m/s \n')
        fd.write(f'Timestep Size:     {delta_t:.4e} seconds\n')
        fd.write('Omega: '+str(omega)+'\n')
        fd.write('ULB:   '+str(settings['ulb'])+'\n\n')  
        fd.write('Results\n')
        fd.write('___________________\n')
        fd.write(f'Time to initialize:   {(time.time()-start_time)/60:.2f} min\n')  
    
  
    solve(
        sim, 
        ulb,
        num_steps, 
        h5exporter, 
        output_dir, 
        grid_shape_zip,
        grid_shape_x_coarsest, 
        delta_x_coarse, 
        shift,
        momentum_transfer,
        reference_area,
        voxel_size,
        prescribed_velocity_phys,
        total_lattice_updates_per_step,
        jsonfile
        )


# Mesh Generation Functions
# =========================
def mesh_prep(voxel_size, car_mesh, body_mesh, wheel_meshes, output_dir, jsonfile):
    
    # Compute bounds on full car
    min_bound = car_mesh.vertices.min(axis=0)
    max_bound = car_mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound  
    
    
    mesher_type = jsonfile['mesher']['type'] 
    # Generate mesh
    if mesher_type == "mres": 
        shift = np.array(
            [               
                jsonfile['mesher']['mres']['domain']["-x"] * partSize[0] - min_bound[0],
                jsonfile['mesher']['mres']['domain']["-y"] * partSize[1] - min_bound[1],
                jsonfile['mesher']['mres']['domain']["-z"] * partSize[2] - min_bound[2],
            ],
            dtype=float,
        ) 
        #Apply shift to car mesh for meshing purpose
        car_mesh.apply_translation(shift)
        _ = car_mesh.vertex_normals
        car_mesh.export("temp.stl")

        # Generate mesh using generate_mesh with ground refinement
        level_data, _, sparsity_pattern, level_origins = generate_mesh(
            jsonfile['mesher']['mres']['levels'],
            "temp.stl",
            jsonfile['settings']['voxelSize'],
            jsonfile['mesher']['mres']['padding'],
            jsonfile['mesher']['mres']['domain'],
            ground_refinement_level=jsonfile['mesher']['mres']['ground_refinement_level'],
            ground_voxel_height=jsonfile['mesher']['mres']['ground_voxel_height'],
        )
    elif mesher_type == "cuboid":  
        # Compute translation to put mesh into first octant of the domain
        domain_multiplier = jsonfile['mesher']['cuboid']
        shift = np.array(
            [
                domain_multiplier[0][0] * partSize[0] - min_bound[0],
                domain_multiplier[0][2] * partSize[1] - min_bound[1],
                domain_multiplier[0][4] * partSize[2] - min_bound[2],
            ],
            dtype=float,
        )
        #Apply shift to car mesh for meshing purpose
        car_mesh.apply_translation(shift)
        _ = car_mesh.vertex_normals
        car_mesh.export("temp.stl")

        # Generate mesh using Cuboid Mesher on full car
        level_data, sparsity_pattern, level_origins = make_cuboid_mesh(
            jsonfile['settings']['voxelSize'],
            domain_multiplier,
            "temp.stl",
        )
    else:
        raise ValueError(f"Invalid mesher_type: {mesher_type}. Must be 'mres' or 'cuboid'.")
   
    # Apply translation to each part 
    body_mesh.apply_translation(shift)
    
    if wheel_meshes is not None:
        wheel_vertices = []
        body_vertices = np.asarray(body_mesh.vertices) / voxel_size
        for mesh in wheel_meshes:    
            mesh.apply_translation(shift)
            if jsonfile['mesher']['trim'] == True:
                zShift = jsonfile['mesher']['trim_voxels']
                plane_origin = np.array([0, 0, mesh.bounds[0][2]+(zShift* voxel_size)])
                plane_normal = np.array([0, 0, 1])  # Upward pointing normal
                # Slice the mesh using the defined plane.
                # With cap=True, the open slice is automatically closed off.
                mesh_above = mesh.slice_plane(plane_origin=plane_origin,
                                  plane_normal=plane_normal,
                                  cap=True)
                mesh_above.export(os.path.join(output_dir, 'temp.stl'))
                wheel_stl = os.path.join(output_dir, 'temp.stl')
                wheel_mesh = trimesh.load_mesh(wheel_stl, process=False)
                wheel_vertices.append(np.asarray(wheel_mesh.vertices) / voxel_size)
    
            else:
                wheel_vertices.append(np.asarray(mesh.vertices) / voxel_size)
    else:
        #No Wheels trim body as needed
        wheel_vertices=None
        if jsonfile['mesher']['trim'] == True:
            zShift = jsonfile['mesher']['trim_voxels']
            plane_origin = np.array([0, 0, body_mesh.bounds[0][2]+(zShift* voxel_size)])
            plane_normal = np.array([0, 0, 1])  # Upward pointing normal
            # Slice the mesh using the defined plane.
            # With cap=True, the open slice is automatically closed off.
            mesh_above = body_mesh.slice_plane(plane_origin=plane_origin,
                        plane_normal=plane_normal,
                        cap=True)
            mesh_above.export(os.path.join(output_dir, 'temp.stl'))
            body_stl = os.path.join(output_dir, 'temp.stl')
            body_mesh = trimesh.load_mesh(body_stl, process=False)
            body_vertices = np.asarray(body_mesh.vertices) / voxel_size
        else:
            body_vertices = np.asarray(body_mesh.vertices) / voxel_size
        

    actual_num_levels = len(level_data)
    grid_shape_finest = tuple([int(i * 2 ** (actual_num_levels - 1)) for i in level_data[-1][0].shape])
    #print(f"Requested levels: {len(domain_multiplier)}, Actual levels: {actual_num_levels}")
    print(f"Full shape based on finest voxel size is {grid_shape_finest}")
    # Clean all temp stls in the folder
    for filename in os.listdir(output_dir):
        # Check if the file ends with '.stl' and is a file (not a directory)
        if filename.endswith('.stl') and os.path.isfile(os.path.join(output_dir, filename)):
            file_path = os.path.join(output_dir, filename)
            os.remove(file_path)
    
    return level_data, body_vertices, wheel_vertices, grid_shape_finest, partSize, actual_num_levels, shift, sparsity_pattern, level_origins

# Boundary Conditions Setup
# =========================
def setup_boundary_conditions(grid, level_data, body_vertices, wheel_vertices, ulb, lbm_visc, grid_shape_zip, precision_policy, jsonfile, velocity_set, compute_backend=ComputeBackend.NEON):
    """
    Set up boundary conditions for the simulation.
    """
    num_levels = len(level_data)
    coarsest_level = num_levels - 1
    box = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level))
    left_indices = grid.boundary_indices_across_levels(level_data, box_side="left", remove_edges=True)
    right_indices = grid.boundary_indices_across_levels(level_data, box_side="right", remove_edges=True)
    top_indices = grid.boundary_indices_across_levels(level_data, box_side="top", remove_edges=False)
    bottom_indices = grid.boundary_indices_across_levels(level_data, box_side="bottom", remove_edges=False)
    front_indices = grid.boundary_indices_across_levels(level_data, box_side="front", remove_edges=False)
    back_indices = grid.boundary_indices_across_levels(level_data, box_side="back", remove_edges=False)

    # Filter front and back indices to remove overlaps with top and bottom at each level
    filtered_front_indices = []
    filtered_back_indices = []
    filtered_top_indices = []
    filtered_bottom_indices = []
    for level in range(num_levels):
        left_set = set(zip(*left_indices[level])) if left_indices[level] else set()
        right_set = set(zip(*right_indices[level])) if right_indices[level] else set()
        top_set = set(zip(*top_indices[level])) if top_indices[level] else set()
        bottom_set = set(zip(*bottom_indices[level])) if bottom_indices[level] else set()
        front_set = set(zip(*front_indices[level])) if front_indices[level] else set()
        back_set = set(zip(*back_indices[level])) if back_indices[level] else set()
        filtered_front_set = front_set - (top_set | bottom_set | left_set | right_set)
        filtered_back_set = back_set - (top_set | bottom_set | left_set | right_set)
        filtered_top_set = top_set - (left_set | right_set)
        filtered_bottom_set = bottom_set - (left_set | right_set)
        filtered_front_indices.append(
            [list(coords) for coords in zip(*filtered_front_set)] if filtered_front_set else []
        )
        filtered_back_indices.append(
            [list(coords) for coords in zip(*filtered_back_set)] if filtered_back_set else []
        )
        filtered_top_indices.append(
            [list(coords) for coords in zip(*filtered_top_set)] if filtered_top_set else []
        )
        filtered_bottom_indices.append(
            [list(coords) for coords in zip(*filtered_bottom_set)] if filtered_bottom_set else []
        )

    # Inlet is either RegularizedBC or Noneq-Reg Hybrid with uniform value (set hybrid if ground refinement is on)
    
    if jsonfile['mesher']['type'] == 'mres' and jsonfile['mesher']['mres']['ground_refinement_level'] > -1 :
        bc_inlet = HybridBC(
            bc_method="nonequilibrium_regularized",
            prescribed_value=(ulb, 0.0, 0.0),
            indices=left_indices,
            )
    elif jsonfile['BCtypes']['inlet'] == "RegularizedBC":
        bc_inlet = RegularizedBC("velocity",
            #profile=bc_profile_new(),
            prescribed_value=(ulb, 0.0, 0.0),
            indices=left_indices,
            )
    else:
        bc_inlet = HybridBC(
            bc_method="nonequilibrium_regularized",
            prescribed_value=(ulb, 0.0, 0.0),
            indices=left_indices,
            )
    
    bc_outlet = DoNothingBC(indices=right_indices)

    # Setup walls moving, static of fall back to FullBounce
    if jsonfile['BCtypes']['walls'] == "moving":
        bc_top =HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(ulb, 0.0, 0.0), indices=top_indices)     
        bc_front =HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(ulb, 0.0, 0.0), indices=filtered_front_indices) 
        bc_back =HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(ulb, 0.0, 0.0), indices=filtered_back_indices)   
    elif jsonfile['BCtypes']['walls'] == "static":
        bc_top =HybridBC(bc_method="nonequilibrium_regularized", indices=top_indices)     
        bc_front =HybridBC(bc_method="nonequilibrium_regularized", indices=filtered_front_indices) 
        bc_back =HybridBC(bc_method="nonequilibrium_regularized", indices=filtered_back_indices)     
    else:
        bc_top = FullwayBounceBackBC(indices=top_indices)     
        bc_front = FullwayBounceBackBC(indices=filtered_front_indices)
        bc_back = FullwayBounceBackBC(indices=filtered_back_indices)

    # Setup ground moving, static or fall back to FullBounce
    if jsonfile['BCtypes']['ground'] == "moving":
        bc_bottom =HybridBC(bc_method="nonequilibrium_regularized", prescribed_value=(ulb, 0.0, 0.0), indices=bottom_indices)   
    elif jsonfile['BCtypes']['ground'] == "static":        
        bc_bottom =HybridBC(bc_method="nonequilibrium_regularized", indices=bottom_indices)     
    else:        
        bc_bottom = FullwayBounceBackBC(indices=bottom_indices)

    # Setup car as grads or non-eq    
    if jsonfile['BCtypes']['car'] == "bounceback_grads":
        bc_body = HybridBC(
            bc_method="bounceback_grads",
            mesh_vertices=body_vertices,
            voxelization_method=MeshVoxelizationMethod("AABB_CLOSE", close_voxels=jsonfile['mesher']['close_voxels']),
            use_mesh_distance=True,
        )
    else:
        bc_body = HybridBC(
            bc_method="nonequilibrium_regularized",
            mesh_vertices=body_vertices,
            voxelization_method=MeshVoxelizationMethod("AABB_CLOSE", close_voxels=jsonfile['mesher']['close_voxels']),
            use_mesh_distance=True,
        )

    # Setup Wheels as rotating or static
    if wheel_vertices is not None:
        wheel_bc = []
        for wheel_vertice in wheel_vertices:
            if jsonfile['BCtypes']['wheels'] == "bounceback_grads":  #need to add movign profile in here somehow
                wheel_bc.append(HybridBC(
                bc_method="bounceback_grads",
                mesh_vertices=wheel_vertice,
                voxelization_method=MeshVoxelizationMethod("AABB_CLOSE", close_voxels=jsonfile['mesher']['close_voxels']),
                use_mesh_distance=True,
                    )
                )
            else:
                wheel_bc.append(HybridBC(
                bc_method="nonequilibrium_regularized",
                mesh_vertices=wheel_vertice,
                voxelization_method=MeshVoxelizationMethod("AABB_CLOSE", close_voxels=jsonfile['mesher']['close_voxels']),
                use_mesh_distance=True,
                    )
                )
        return wheel_bc + [bc_top, bc_bottom, bc_front, bc_back, bc_inlet, bc_outlet, bc_body] # Body must be last. Outlet must be second to last
    else:
        return [bc_top, bc_bottom, bc_front, bc_back, bc_inlet, bc_outlet, bc_body] # Body must be last. Outlet must be second to last

# Utility Functions
# =================
def print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size, drag_values):
    """
    Calculate and print lift and drag coefficients.
    """
    boundary_force = momentum_transfer(sim.f_0, sim.f_1, sim.bc_mask, sim.missing_mask)
    drag = boundary_force[0]
    lift = boundary_force[2]
    cd = 2.0 * drag / (ulb**2 * reference_area)
    cl = 2.0 * lift / (ulb**2 * reference_area)
    if np.isnan(cd) or np.isnan(cl):
        raise ValueError(f"NaN detected in coefficients at step {step}: Cd={cd}, Cl={cl}")
    drag_values.append([cd, cl])
    # print(f"CD={cd:.3f}, CL={cl:.3f}, Drag Force (lattice units)={drag:.6f}")
    return cd, cl, drag

def plot_drag_lift(drag_values, output_dir, print_interval, script_name, percentile_range=(15, 85), use_log_scale=False):
    """
    Plot CD and CL over time and save the plot to the output directory.
    """
    drag_values_array = np.array(drag_values)
    steps = np.arange(0, len(drag_values) * print_interval, print_interval)
    cd_values = drag_values_array[:, 0]
    cl_values = drag_values_array[:, 1]
    y_min = min(np.percentile(cd_values, percentile_range[0]), np.percentile(cl_values, percentile_range[0]))
    y_max = max(np.percentile(cd_values, percentile_range[1]), np.percentile(cl_values, percentile_range[1]))
    padding = (y_max - y_min) * 0.1
    y_min, y_max = y_min - padding, y_max + padding
    if use_log_scale:
        y_min = max(y_min, 1e-6)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cd_values, label='Drag Coefficient (Cd)', color='blue')
    plt.plot(steps, cl_values, label='Lift Coefficient (Cl)', color='red')
    plt.xlabel('Simulation Step')
    plt.ylabel('Coefficient')
    plt.title(f'{script_name}: Drag and Lift Coefficients Over Time')
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    if use_log_scale:
        plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'drag_lift_plot.png'))
    plt.close()

def compute_voxel_statistics_and_reference_area(sim, bc_mask_exporter, level_data, actual_num_levels, sparsity_pattern, boundary_conditions, voxel_size):
    """
    Compute active/solid voxels, totals, lattice updates, and reference area based on simulation data.
    """
    # Compute macro fields
    sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
    fields_data = bc_mask_exporter.get_fields_data({"bc_mask": sim.bc_mask})
    bc_mask_data = fields_data["bc_mask_0"]
    level_id_field = bc_mask_exporter.level_id_field

    # Compute solid voxels per level (assuming 255 is the solid marker)
    solid_voxels = []
    for lvl in range(actual_num_levels):
        level_mask = level_id_field == lvl
        solid_voxels.append(np.sum(bc_mask_data[level_mask] == 255))

    # Compute active voxels (total non-zero in sparsity minus solids)
    active_voxels = [np.count_nonzero(mask) for mask in sparsity_pattern]
    active_voxels = [max(0, active_voxels[lvl] - solid_voxels[lvl]) for lvl in range(actual_num_levels)]

    # Totals
    total_voxels = sum(active_voxels)
    total_lattice_updates_per_step = sum(active_voxels[lvl] * (2 ** (actual_num_levels - 1 - lvl)) for lvl in range(actual_num_levels))

    # Compute reference area (projected on YZ plane at finest level)
    finest_level = 0
    mask_finest = level_id_field == finest_level
    bc_mask_finest = bc_mask_data[mask_finest]
    active_indices_finest = np.argwhere(level_data[0][0])
    bc_body_id = boundary_conditions[-1].id  # Assuming last BC is bc_body
    solid_voxels_indices = active_indices_finest[bc_mask_finest == bc_body_id]
    unique_jk = np.unique(solid_voxels_indices[:, 1:3], axis=0)
    reference_area = unique_jk.shape[0]
    reference_area_physical = reference_area * (voxel_size ** 2)

    return {
        "active_voxels": active_voxels,
        "solid_voxels": solid_voxels,
        "total_voxels": total_voxels,
        "total_lattice_updates_per_step": total_lattice_updates_per_step,
        "reference_area": reference_area,
        "reference_area_physical": reference_area_physical
    }


def save_slices(output_dir, grid_shape_zip, shift, h5exporter, delta_x_coarse,voxel_size, sim,jsonfile):
    domainSize = np.array(grid_shape_zip) * voxel_size
    outputSlices = jsonfile['outputSlices']
    # Map axis to plane normal
    axis_to_normal = {
        'X': [1, 0, 0],
        'Y': [0, 1, 0],
        'Z': [0, 0, 1]
    }
    domain_min = -shift
    domain_max = domain_min + domainSize
    tic = time.time()
    def compute_slice_bounds_relative_to_domain(origin, width, height, width_vec, height_vec, domain_min, domain_max, plane_normal):        
        # Build in-plane basis
        n = np.array(plane_normal) / np.linalg.norm(plane_normal)
        if np.allclose(n, [1, 0, 0]):
            u1 = np.array([0, 1, 0])
        else:
            u1 = np.array([1, 0, 0])
        u1 = u1 / np.linalg.norm(u1)
        u2 = np.cross(n, u1)
        u2 = u2 / np.linalg.norm(u2)
        width_vec_norm = width_vec / np.linalg.norm(width_vec)
        height_vec_norm = height_vec / np.linalg.norm(height_vec)
        if np.dot(u1, width_vec_norm) < 0:
            u1 = -u1
        if np.dot(u2, height_vec_norm) < 0:
            u2 = -u2

        # Use the lower-left corner of the slice as the reference point
        ref_point = origin

        # Project domain corners onto the plane and compute in-plane coordinates
        domain_corners = np.array([
            [domain_min[0], domain_min[1], domain_min[2]],
            [domain_max[0], domain_min[1], domain_min[2]],
            [domain_min[0], domain_max[1], domain_min[2]],
            [domain_max[0], domain_max[1], domain_min[2]],
            [domain_min[0], domain_min[1], domain_max[2]],
            [domain_max[0], domain_min[1], domain_max[2]],
            [domain_min[0], domain_max[1], domain_max[2]],
            [domain_max[0], domain_max[1], domain_max[2]]
        ])
        local_corners = []
        for corner in domain_corners:
            # Project corner onto the plane
            proj = corner - np.dot(corner - ref_point, n) * n
            local_x = np.dot(proj - ref_point, u1)
            local_y = np.dot(proj - ref_point, u2)
            local_corners.append([local_x, local_y])
        local_corners = np.array(local_corners)
        domain_u_min, domain_u_max = local_corners[:, 0].min(), local_corners[:, 0].max()
        domain_v_min, domain_v_max = local_corners[:, 1].min(), local_corners[:, 1].max()

        # Project slice corners onto the plane and compute in-plane coordinates
        slice_corners = [
            origin,
            origin + width * width_vec,
            origin + height * height_vec,
            origin + width * width_vec + height * height_vec
        ]
        slice_local = []
        for corner in slice_corners:
            proj = corner - np.dot(corner - ref_point, n) * n
            local_x = np.dot(proj - ref_point, u1)
            local_y = np.dot(proj - ref_point, u2)
            slice_local.append([local_x, local_y])
        slice_local = np.array(slice_local)
        slice_u_min, slice_u_max = slice_local[:, 0].min(), slice_local[:, 0].max()
        slice_v_min, slice_v_max = slice_local[:, 1].min(), slice_local[:, 1].max()

        # Convert to fractions
        u_min = (slice_u_min - domain_u_min) / (domain_u_max - domain_u_min)
        u_max = (slice_u_max - domain_u_min) / (domain_u_max - domain_u_min)
        v_min = (slice_v_min - domain_v_min) / (domain_v_max - domain_v_min)
        v_max = (slice_v_max - domain_v_min) / (domain_v_max - domain_v_min)

        return [max(0,u_min), min(1,u_max), max(0,v_min), min(1,v_max)]
    
    for slice_group in outputSlices:
        field_name = slice_group['field']
        axis = slice_group['axis']
        height = slice_group['height']
        width = slice_group['width']
    # Extract vectors
        height_vec = np.array([
            slice_group['heightVec']['x'],
            slice_group['heightVec']['y'],
            slice_group['heightVec']['z']
        ])
        width_vec = np.array([
            slice_group['widthVec']['x'],
            slice_group['widthVec']['y'],
            slice_group['widthVec']['z']
        ])
        
        # Get plane normal
        plane_normal = axis_to_normal[axis]
        
        # Process each origin
        for idx, origin_dict in enumerate(slice_group['origin']):
            # The origin / plane point is the lower-left corner of the slice
            plane_point = np.array([
                origin_dict['x'],
                origin_dict['y'],
                origin_dict['z']
            ])
            
            # Calculate bounds in model units
            
            # Calculate the bounds
            # Since we're given absolute dimensions, we need to compute
            # the bounds relative to the full domain extent in the plane
            # For now, we'll use bounds [0, 1, 0, 1] to capture the full slice
            # as defined by the width and height
            bounds = [0, 1, 0, 1]
            bounds_x, bounds_x2, bounds_y, bounds_y2 = compute_slice_bounds_relative_to_domain(plane_point, width, height, width_vec, height_vec, domain_min, domain_max, plane_normal)
            print(f'bounds {bounds_x}, {bounds_x2}, {bounds_y}, {bounds_y2}')
            # Alternatively, if you want to compute bounds relative to domain:
            # You would need to:
            # 1. Project domain extents onto the plane
            # 2. Calculate where this slice sits within those extents
            # 3. Set bounds accordingly
            
            # Generate output filename
            output_filename = os.path.join(
                output_dir,
                f"{axis}_slice_{idx:03d}"
            )
            
            print(f"Generating slice: {output_filename}")
            print(f"  Axis: {axis}, Normal: {plane_normal}")
            print(f"  Plane point: {plane_point}")
            print(f"  Width: {width}, Height: {height}")
            wp.synchronize()
            print(f"Max Velocity for slice scaling: {jsonfile['InletBC']['x'] * jsonfile['settings']['sliceFactor']}")
            h5exporter.to_slice_image(
                output_filename,
                {"velocity": sim.u},
                plane_point=plane_point,
                plane_normal=plane_normal,
                grid_res=jsonfile['settings']['grid_res'],
                bounds=(bounds_x, bounds_x2, bounds_y, bounds_y2),
                show_axes=False,
                show_colorbar=False,
                cmap=jsonfile['settings']['sliceColorMap'],
                normalize=jsonfile['InletBC']['x'] * jsonfile['settings']['sliceFactor'],
                slice_thickness=delta_x_coarse #needed when using model units
            )
    print(f"Time to save all images {time.time()-tic} seconds. ")
  
  
def solve(
        sim, 
        ulb,
        num_steps, 
        h5exporter, 
        output_dir, 
        grid_shape_zip,
        grid_shape_x_coarsest, 
        delta_x_coarse, 
        shift,
        momentum_transfer,
        reference_area,
        voxel_size,
        prescribed_velocity_phys,
        total_lattice_updates_per_step,
        jsonfile
        ):
    
    # -------------------------- Simulation Loop --------------------------
    wp.synchronize()
    print(f"\n*******\nSolver Started\n*******\n")
    start_time = time.time()
    solve_start = start_time
    compute_time = 0.0
    steps_since_last_print = 0
    drag_values = []
    
    # Calculate print and file output intervals
    print_interval = max(1, int(num_steps * (jsonfile['settings']['solutionPrintFreq'] / 100.0)))
    crossover_step = int(num_steps * (jsonfile['settings']['crossover'] / 100.0))
    file_output_interval_pre_crossover = max(1, int(crossover_step / jsonfile['settings']['preCrossover_frames'])) if jsonfile['settings']['preCrossover_frames'] > 0 else num_steps + 1
    file_output_interval_post_crossover = max(1, int((num_steps - crossover_step) / jsonfile['settings']['postCrossover_frames'])) if jsonfile['settings']['postCrossover_frames'] > 0 else num_steps + 1
    final_print_interval = max(1, int((num_steps-crossover_step) * (jsonfile['settings']['crossover'] / 100.0)))

    if jsonfile['settings']['debug']:
        for step in range(num_steps):
            solution_time =(time.time()-solve_start)/60
            step_start = time.time()
            sim.step()
            wp.synchronize()
            compute_time += time.time() - step_start
            steps_since_last_print += 1
            percent_complete = (step + 1) / num_steps * 100
            scm_progress(np.floor(percent_complete))
            end_time = time.time()
            elapsed = end_time - start_time
            time_out = False
            if elapsed/60 >= jsonfile['settings']['limit']:
                time_out = True
            if (step % print_interval == 0 and step < crossover_step) or step == num_steps - 1 or time_out:
                sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
                wp.synchronize()
                cd, cl, drag = print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size, drag_values)
                filename = os.path.join(output_dir, f"{jsonfile['outputName']}_{step:04d}")
                h5exporter.to_slice_image(
                    filename,
                    {"velocity": sim.u},
                    plane_point=(1, 0, 0),
                    plane_normal=(0, 1, 0),
                    grid_res=jsonfile['settings']['grid_res'],
                    bounds=(0.25, 0.75, 0, 0.5),
                    show_axes=False,
                    show_colorbar=False,
                    slice_thickness=delta_x_coarse, #needed when using model units
                    normalize = prescribed_velocity_phys*jsonfile['settings']['sliceFactor'], 
                )
                
                total_lattice_updates = total_lattice_updates_per_step * steps_since_last_print
                MLUPS = total_lattice_updates / compute_time / 1e6 if compute_time > 0 else 0.0
                current_flow_passes = step * ulb / grid_shape_x_coarsest
                remaining_steps = num_steps - step - 1
                time_remaining = 0.0 if MLUPS == 0 else (total_lattice_updates_per_step * remaining_steps) / (MLUPS * 1e6)
                hours, rem = divmod(time_remaining, 3600)
                minutes, seconds = divmod(rem, 60)
                time_remaining_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
                
                print(f"Completed step {step}/{num_steps} ({percent_complete:.2f}% complete)")
                print(f"  Flow Passes: {current_flow_passes:.2f}")
                print(f"  Time elapsed: {elapsed:.1f}s, Compute time: {compute_time:.1f}s, ETA: {time_remaining_str}")
                print(f"  MLUPS: {MLUPS:.1f}")
                print(f"  Cd={cd:.3f}, Cl={cl:.3f}, Drag Force (lattice units)={drag:.3f}")
                #start_time = time.time()
                compute_time = 0.0
                steps_since_last_print = 0
                scm_results_available() 
                if time_out:
                    wp.synchronize()
                    sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
                    save_slices(output_dir, grid_shape_zip, shift, h5exporter, delta_x_coarse,voxel_size, sim,jsonfile)                                        
                    filename = os.path.join(output_dir, f"{jsonfile['outputName']}_{step:04d}")
                    h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=1)

                    with open(os.path.join(output_dir, "project.log"),'a') as fd:
                        fd.write(f"*** Solution Timed out ***\n")
                        fd.write(f"Actual iterations: {step}\n")
                    print('Time limit reached')
                    break
                
            file_output_interval = file_output_interval_pre_crossover if step < crossover_step else file_output_interval_post_crossover
            if step % file_output_interval == 0 or step == num_steps - 1:
                sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
                filename = os.path.join(output_dir, f"{jsonfile['outputName']}_{step:04d}")
                h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=1)
                wp.synchronize()
            if step >= crossover_step and step % final_print_interval ==0 :
                sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
                wp.synchronize()
                cd, cl, drag = print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size)
                print(f"Completed step {step}/{num_steps} ")
                print(f"  Cd= {cd:.3f}, Cl= {cl:.3f}, Drag Force (lattice units)={drag:.3f}")
                filename = os.path.join(output_dir, f"{jsonfile['outputName']}_{step:04d}")
                h5exporter.to_slice_image(
                    filename,
                    {"velocity": sim.u},
                    plane_point=(1, 0, 0),
                    plane_normal=(0, 1, 0),
                    grid_res=jsonfile['settings']['grid_res'],
                    bounds=(0, 1, 0, 1),
                    show_axes=False,
                    show_colorbar=False,
                    slice_thickness=delta_x_coarse, #needed when using model units
                    normalize = prescribed_velocity_phys*jsonfile['settings']['sliceFactor'], 
                )
                

            

                
        # Save drag and lift data to CSV
        if len(drag_values) > 0:
            with open(os.path.join(output_dir, "drag_lift.csv"), 'w') as fd:
                fd.write("Step,Cd,Cl\n")
                for i, (cd, cl) in enumerate(drag_values):
                    fd.write(f"{i * print_interval},{cd},{cl}\n")
            plot_drag_lift(drag_values, output_dir, print_interval, jsonfile['outputName'])

            # Calculate and print average Cd and Cl for the last 50%
            drag_values_array = np.array(drag_values)
        
            start_index = int(len(drag_values) * (jsonfile['settings']['crossover'] / 100.0))        
            last_half = drag_values_array[start_index:, :]
            avg_cd = np.mean(last_half[:, 0])
            avg_cl = np.mean(last_half[:, 1])
            epsilon = 1e-6
            target_cd = jsonfile['vehicle']['targets']['cd'] + epsilon
            target_cl = jsonfile['vehicle']['targets']['cl'] + epsilon
            print(f"Experimental Drag Coefficient (Cd): {target_cd}\n" 
                f"Average Drag Coefficient (Cd) for last {100-jsonfile['settings']['crossover']}%: {avg_cd:.6f}\n"
                f"Average Lift Coefficient (Cl) for last {100-jsonfile['settings']['crossover']}%: {avg_cl:.6f}\n"
                f"Error Drag Coefficient (Cd): {((avg_cd-target_cd)/target_cd)*100:.2f}%\n" 
                f"Error Lift Coefficient (Cl): {((avg_cl-target_cl)/target_cl)*100:.2f}%\n"
                )
            
            with open(os.path.join(output_dir, "project.log"),'a') as fd:
                fd.write(f"Average Drag Coefficient (Cd) for last {100-jsonfile['settings']['crossover']}%: {avg_cd:.6f}\n")
                fd.write(f"Average Lift Coefficient (Cl) for last {100-jsonfile['settings']['crossover']}%: {avg_cl:.6f}\n")
                fd.write(f"Error Drag Coefficient (Cd): {((avg_cd-target_cd)/target_cd)*100:.2f}%\n")
                fd.write(f"Error Lift Coefficient (Cl): {((avg_cl-target_cl)/target_cl)*100:.2f}%\n")
                fd.write(f'Total Solution Time:     {(time.time()-solve_start)/60:.3f} min\n')
                
        save_slices(output_dir, grid_shape_zip, shift, h5exporter, delta_x_coarse, voxel_size,sim,jsonfile)          
        with open(os.path.join(output_dir, "source.json"), 'w') as file:
            json.dump(jsonfile, file, indent=4) # indent for pretty-printing
            print(f"Source Json written to {os.path.join(output_dir, 'source.json')} successfully.")
            
        scm_results_available(True)
    # Customer style run (no extra debug outputs)
    # Runs setup and then only takes data from crossover to end
    else:
        print_interval=max(1, int((num_steps-crossover_step) * (jsonfile['settings']['solutionPrintFreq'] / 100.0)))

        for step in range(num_steps):
            end_time = time.time()
            elapsed = end_time - start_time
            sim.step()
            wp.synchronize()
            percent_complete = (step + 1) / num_steps * 100
            scm_progress(np.floor(percent_complete))
            
            if elapsed/60 >= jsonfile['settings']['limit']:
                sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
                wp.synchronize()
                cd, cl, drag = print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size, drag_values)              
                
                with open(os.path.join(output_dir, "project.log"),'a') as fd:
                    fd.write(f"*** Solution Time Reached ***\n")
                    fd.write(f"Actual iterations: {step}\n")
                print('Time limit reached')
                save_slices(output_dir, grid_shape_zip, shift, h5exporter, delta_x_coarse, voxel_size,sim,jsonfile)  
                if jsonfile['settings']['fullData']==True:
                    wp.synchronize()
                    sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
                    filename = os.path.join(output_dir, f"{jsonfile['outputName']}_{step:04d}")
                    h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=1)
      
                scm_results_available() 
                break
                    
            if step >= crossover_step:            
                if step % print_interval == 0 or step == num_steps - 1:
                    sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
                    wp.synchronize()
                    cd, cl, drag = print_lift_drag(sim, step, momentum_transfer, ulb, reference_area, voxel_size, drag_values)              
                    scm_results_available() 
                    print(f"Completed step {step}/{num_steps}")
                    print(f"  Cd={cd:.3f}, Cl={cl:.3f}, Drag Force (lattice units)={drag:.3f}")
                
            if (step == num_steps - 1) & (jsonfile['settings']['fullData']==True):            
                wp.synchronize()
                sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)
                filename = os.path.join(output_dir, f"{jsonfile['outputName']}_{step:04d}")
                h5exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=0)
      
        # Save drag and lift data to CSV
        if len(drag_values) > 0:
            with open(os.path.join(output_dir, "drag_lift.csv"), 'w') as fd:
                fd.write("Step,Cd,Cl\n")
                for i, (cd, cl) in enumerate(drag_values):
                    fd.write(f"{i * print_interval},{cd},{cl}\n")
            plot_drag_lift(drag_values, output_dir, print_interval, jsonfile['outputName'])

            # Calculate and print average Cd and Cl for the last 50%
            drag_values_array = np.array(drag_values)
        
            #start_index = int(len(drag_values) * (jsonfile['settings']['crossover'] / 100.0))        
            #last_half = drag_values_array[start_index:, :]
            avg_cd = np.mean(drag_values_array[:, 0])
            avg_cl = np.mean(drag_values_array[:, 1])
            epsilon = 1e-6
            target_cd = jsonfile['vehicle']['targets']['cd'] + epsilon
            target_cl = jsonfile['vehicle']['targets']['cl'] + epsilon
            print(f"Experimental Drag Coefficient (Cd): {0.307}\n" 
            f"Average Drag Coefficient (Cd) for last {100-jsonfile['settings']['crossover']}%: {avg_cd:.6f}\n"
            f"Average Lift Coefficient (Cl) for last {100-jsonfile['settings']['crossover']}%: {avg_cl:.6f}\n"
            f"Error Drag Coefficient (Cd): {((avg_cd-target_cd)/target_cd)*100:.2f}%\n" 
            f"Error Lift Coefficient (Cl): {((avg_cl-target_cl)/target_cl)*100:.2f}%\n"
            )
            
            with open(os.path.join(output_dir, "project.log"),'a') as fd:
                fd.write(f"Average Drag Coefficient (Cd) for last {100-jsonfile['settings']['crossover']}%: {avg_cd:.6f}\n")
                fd.write(f"Average Lift Coefficient (Cl) for last {100-jsonfile['settings']['crossover']}%: {avg_cl:.6f}\n")
                fd.write(f"Error Drag Coefficient (Cd): {((avg_cd-target_cd)/target_cd)*100:.2f}%\n")
                fd.write(f"Error Drag Coefficient (Cl): {((avg_cl-target_cl)/target_cl)*100:.2f}%\n")
                fd.write(f'Total Solution Time:     {(time.time()-solve_start)/60:.3f} min\n')
        save_slices(output_dir, grid_shape_zip, shift, h5exporter, delta_x_coarse,voxel_size, sim,jsonfile)          
            
        scm_results_available(True)



def main(argv):
    """
    Main entry point for the Studio Wind Tunnel Solver.

    Parses command-line arguments to obtain the input JSON file, initializes the simulation environment,
    cleans up previous output files, and runs the wind tunnel simulation. Handles errors and reports
    progress and completion status via SCM events.

    Args:
        argv (list): List of command-line arguments.

    Returns:
        int: Exit code. Returns 0 on success, 64 on argument/input errors, or 1 on simulation failure.
    """

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    input_file = ''
    usage = 'windtunnel_json.py -i <inputjson>'

    logging.info('Welcome to Studio Wind Tunnel Solver')  

    try:
        opts, _ = getopt.getopt(argv, "hi:o:", ["ifile="])
    except getopt.GetoptError:
        logging.error(usage)
        scm_set_error(64, 'Argument error')
        return 64

    for opt, arg in opts:
        if opt == '-h':
            logging.info(usage)
            return 64

        if opt in ("-i", "--ifile"):
            input_file = arg

    if not input_file:
        logging.error('Error: Input JSON file must be specified.\n' + usage)
        scm_set_error(64, 'Input file not specified')
        return 64

    try:
        if running_via_scm():
            log_file_scm = os.path.join(os.path.dirname(os.path.abspath(input_file)), 'solve.log')
            scm_log_handler = logging.FileHandler(log_file_scm, mode='w')
            scm_log_handler.setLevel(logging.INFO)
            scm_log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
            logging.getLogger().addHandler(scm_log_handler)
            logging.info('SCM Log file: {}'.format(log_file_scm))

        logging.info('Input file: {}'.format(input_file))

        scm_init()
        
        prep_inputs(input_file)

        scm_complete()
    except Exception as e:
        logging.error(f'Exception occured: {e}')
        scm_set_error(1, f'Job failed: {e}')
        scm_cancel_heartbeat()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
