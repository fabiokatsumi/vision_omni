$scriptFolder = "\\host.lan\Data"
$pythonScriptFile = "$scriptFolder\server\main.py"
$pythonServerPort = 5006

# Start the flask computer use server
Write-Host "Running the server on port $pythonServerPort"
python $pythonScriptFile --port $pythonServerPort
