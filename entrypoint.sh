#!/bin/bash
set -e

echo "==================================="
echo "STT Gateway Docker Container"
echo "==================================="

# Create necessary directories
mkdir -p logs debug_audio models

# Parse command - handle both direct calls and calls from NVIDIA entrypoint
# If no arguments provided, default to "all"
if [ $# -eq 0 ]; then
    SERVICE="all"
else
    SERVICE="$1"
fi

# Function to start a service in background
start_service() {
    local name=$1
    local cmd=$2
    echo "Starting $name..."
    $cmd > "logs/${name}.log" 2>&1 &
    local pid=$!
    echo "$pid" > "logs/${name}.pid"
    echo "  $name started with PID: $pid"
}

# Extract port from URL (assumes http://host:port format)
extract_port() {
    echo "$1" | grep -oE '[0-9]+$'
}

# Set default worker URLs if not already set
export ASR_PARAKEET_WORKER_URL="${ASR_PARAKEET_WORKER_URL:-http://127.0.0.1:8093}"
export ASR_MOONSHINE_WORKER_URL="${ASR_MOONSHINE_WORKER_URL:-http://127.0.0.1:8094}"

echo "Configuration:"
echo "  ASR_PARAKEET_WORKER_URL: $ASR_PARAKEET_WORKER_URL"
echo "  ASR_MOONSHINE_WORKER_URL: $ASR_MOONSHINE_WORKER_URL"
echo ""

# Start services based on command
# Function to start ASR workers
start_asr_workers() {
    echo "Starting ASR Workers..."
    
    # Start Parakeet worker
    local parakeet_port=$(extract_port "$ASR_PARAKEET_WORKER_URL")
    if [ -n "$parakeet_port" ]; then
        start_service "parakeet_worker" "python3 src/asr_parakeet_worker_app.py --port $parakeet_port"
    fi
    
    # Start Moonshine worker
    local moonshine_port=$(extract_port "$ASR_MOONSHINE_WORKER_URL")
    if [ -n "$moonshine_port" ]; then
        start_service "moonshine_worker" "python3 src/asr_moonshine_worker_app.py --port $moonshine_port"
    fi
}

case "$SERVICE" in
    all)
        echo "Starting all services..."
        
        # Start Speaker Verification Service
        start_service "speaker_verification" "python3 src/speaker_verification_app.py"
        sleep 2
        
        # Start Smart Turn Worker
        start_service "smart_turn_worker" "python3 src/smart_turn_worker_app.py --port ${SMART_TURN_WORKER_PORT:-8091}"
        sleep 2
        
        # Start ASR Workers
        start_asr_workers
        
        # Wait for workers to be ready
        echo ""
        echo "Waiting for workers to initialize..."
        sleep 10
        
        # Start ASR Batching Worker
        start_service "asr_batching_worker" "python3 src/asr_batching_worker_app.py"
        sleep 3
        
        # Start Gateway (in foreground)
        echo ""
        echo "Starting Gateway Service (foreground)..."
        echo "==================================="
        exec python3 src/gateway_app.py
        ;;
        
    gateway)
        echo "Starting Gateway only..."
        exec python3 src/gateway_app.py
        ;;
        
    workers)
        echo "Starting worker services only..."
        start_service "speaker_verification" "python3 src/speaker_verification_app.py"
        start_service "smart_turn_worker" "python3 src/smart_turn_worker_app.py --port ${SMART_TURN_WORKER_PORT:-8091}"
        
        # Start ASR Workers
        start_asr_workers
        
        sleep 5
        start_service "asr_batching_worker" "python3 src/asr_batching_worker_app.py"
        
        # Keep container running
        echo "Workers started. Tailing logs..."
        tail -f logs/*.log
        ;;
        
    *)
        echo "Unknown command: $SERVICE"
        echo "Usage: $0 {all|gateway|workers}"
        exit 1
        ;;
esac

