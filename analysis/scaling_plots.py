import matplotlib.pyplot as plt

def plot_scaling(process_counts, speedups):
    plt.figure()
    plt.plot(process_counts, speedups, marker='o')
    plt.xlabel("Number of Processes / Devices")
    plt.ylabel("Speedup")
    plt.title("Strong Scaling Curve")
    plt.grid(True)
    plt.savefig("scaling_curve.png")
    plt.show()

# Example usage:
# plot_scaling([1,2,4,8], [1,1.8,3.5,6.9])
