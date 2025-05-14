nohup bash -c '
list_of_names=("PHASE1A" "PHASE1B" "PHASE2A" "PHASE2B")
for name in "${list_of_names[@]}"; do
  python examples/cfd/building_ventilation.py "$name"
done
' > nohup_all.out 2>&1 &