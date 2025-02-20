import os  # Import the os module for interacting with the operating system
import json  # Import the json module for working with JSON data
import zipfile  # Import the zipfile module for working with ZIP files
import subprocess  # Import the subprocess module for running subprocesses
import logging  # Import the logging module for logging messages
import sys  # Import the sys module for system-specific parameters and functions
from kaggle.api.kaggle_api_extended import KaggleApi  # Import the KaggleApi class from the kaggle package
from tkinter import Tk  # Import the Tk class from the tkinter module
from tkinter.filedialog import askopenfilename  # Import the askopenfilename function from the tkinter.filedialog module

# Configure logging to display info level messages
logging.basicConfig(level=logging.INFO)

# Check if Kaggle package is installed, if not install it
try:
    import kaggle  # Try to import the kaggle package
except ImportError:  # If the kaggle package is not installed
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])  # Install the kaggle package using pip

# Load Kaggle credentials from kaggle.json file
kaggle_json_path = os.path.join(os.getcwd(), "kaggle.json")
if not os.path.exists(kaggle_json_path):
    logging.info("kaggle.json file not found. Please select the kaggle.json file.")
    Tk().withdraw()  # Hide the root window
    kaggle_json_path = askopenfilename(title="Select kaggle.json file", filetypes=[("JSON files", "*.json")])
    if not kaggle_json_path:
        logging.error("No kaggle.json file selected.")
        sys.exit(1)

try:
    with open(kaggle_json_path, "r") as f:  # Open the kaggle.json file for reading
        kaggle_credentials = json.load(f)  # Load the credentials from the file
except IOError as e:  # Handle any IO errors that occur
    logging.error(f"Failed to read kaggle.json: {e}")
    sys.exit(1)

# Check if Kaggle credentials are set, if not log an error and exit
if not kaggle_credentials.get("username") or not kaggle_credentials.get("key"):
    logging.error("Kaggle credentials are not set in kaggle.json.")
    sys.exit(1)

# Create the .kaggle directory in the user's home directory if it doesn't exist
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# Write the Kaggle credentials to the kaggle.json file in the .kaggle directory
kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
try:
    with open(kaggle_json_path, "w") as f:  # Open the kaggle.json file for writing
        json.dump(kaggle_credentials, f)  # Write the credentials to the file in JSON format
    os.chmod(kaggle_json_path, 0o600)  # Set the file permissions to read/write for the owner only
except IOError as e:  # Handle any IO errors that occur
    logging.error(f"Failed to write kaggle.json: {e}")
    sys.exit(1)

# Authenticate with the Kaggle API
api = KaggleApi()
api.authenticate()

# Prompt the user to enter the Kaggle competition command
competition_command = input("Please enter the Kaggle competition command (e.g., 'kaggle competitions download -c march-machine-learning-mania-2025'): ")

# Extract the competition name from the command
try:
    competition_name = competition_command.split("-c")[1].strip()
except IndexError:
    logging.error("Invalid competition command format.")
    sys.exit(1)

# Download the competition files for the specified competition
try:
    api.competition_download_files(competition_name, path='.')
except Exception as e:  # Handle any exceptions that occur during download
    logging.error(f"Failed to download competition files: {e}")
    sys.exit(1)

# Define the path to the downloaded ZIP file
zip_file_path = f"{competition_name}.zip"

# Check if the ZIP file exists
if os.path.exists(zip_file_path):
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:  # Open the ZIP file for reading
            zip_ref.extractall(competition_name)  # Extract all contents to the specified directory
    except zipfile.BadZipFile as e:  # Handle any errors that occur during extraction
        logging.error(f"Failed to unzip file: {e}")
else:
    logging.error(f"File {zip_file_path} does not exist. Please check the file path.")  # Log an error if the file does not exist