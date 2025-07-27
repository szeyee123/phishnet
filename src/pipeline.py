import subprocess
import time

# Preprocessing
def run_preprocessing():
    print("[1/5] Running preprocessing.py...")
    subprocess.run(["python", "preprocessing.py"], check=True)

# Random Forest Training
def run_rf_train():
    print("[2/5] Running rf_train.py...")
    subprocess.run(["python", "rf_train.py"], check=True)

# KMeans Clustering Training
def run_kmeans_train():
    print("[3/5] Running kmeans_train.py...")
    subprocess.run(["python", "kmeans_train.py"], check=True)
    subprocess.run(["python", "megan_confusionmatrix.py"], check=True)
    subprocess.run(["python", "megan_averagevalue.py"], check=True)

# Autoencoder Training
def run_autoencoder_train():
    print("[4/5] Running autoencoder_train.py...")
    subprocess.run(["python", "autoencoder_train.py"], check=True)

# Combined Predictions
def run_combined_predictions():
    print("[5/5] Running combined_predict.py...")
    subprocess.run(["python", "combined_predict.py"], check=True)

# Launch Dashboard after training
def launch_dashboard():
    print("Launching dashboard.py...")
    subprocess.run(["python", "dashboard.py"])

if __name__ == "__main__":
    start_time = time.time()

    try:
        run_preprocessing()
        run_rf_train()
        run_kmeans_train()
        run_autoencoder_train()
        run_combined_predictions()

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