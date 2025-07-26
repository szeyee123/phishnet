import subprocess
import time

# Preprocessing
def run_preprocessing():
    print("[1/4] Running preprocessing.py...")
    subprocess.run(["python", "preprocessing.py"], check=True)

# Random Forest Training
def run_rf_train():
    print("[2/4] Running rf_train.py...")
    subprocess.run(["python", "rf_train.py"], check=True)

# KMeans Clustering Training


# Autoencoder Training


# Launch Dashboard after training
def launch_dashboard():
    print("Launching dashboard.py...")
    subprocess.run(["python", "dashboard.py"])

if __name__ == "__main__":
    start_time = time.time()

    try:
        run_preprocessing()
        run_rf_train()

        duration = time.time() - start_time
        print(f"Pipeline completed in {duration:.2f} seconds.")

        # Prompt user to launch dashboard
        launch = input("Do you want to launch the dashboard now? (y/n): ")
        if launch.lower() == 'y':
            launch_dashboard()
        else:
            print("Dashboard launch skipped.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")