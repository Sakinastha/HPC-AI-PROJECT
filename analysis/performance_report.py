def create_report(results_dict):
    print("===== HPC Deep Learning Performance Report =====")
    for key, value in results_dict.items():
        print(f"{key}: {value}")

# Example usage:
# results = {"Serial": "120s", "MPI(4)": "35s", "MPS": "20s", "Hybrid": "12s"}
# create_report(results)
