#!/usr/bin/env python3
"""
FastAPI web server for robust_tiff_compress.

Provides a web interface to start, monitor, and stop TIFF compression runs.
"""

import os
import sys
import threading
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Add parent directory to path to import robust_tiff_compress
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robust_tiff_compress import run_compression


class LogCaptureHandler(logging.Handler):
    """Custom logging handler that captures logs to a list and optionally to a file."""
    
    def __init__(self, log_store: List[str], max_lines: int = 2000, log_file_path: Optional[str] = None):
        super().__init__()
        self.log_store = log_store
        self.max_lines = max_lines
        self.log_file_path = log_file_path
        self.log_file = None
        self._file_lock = threading.Lock()
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                           datefmt='%Y-%m-%d %H:%M:%S'))
        
        # Open log file if path is provided
        if self.log_file_path:
            try:
                # Ensure directory exists
                log_dir = os.path.dirname(self.log_file_path)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                # Open file in append mode
                self.log_file = open(self.log_file_path, 'a', encoding='utf-8')
                # Change permissions to be read+write for group and others (0666)
                try:
                    os.chmod(self.log_file_path, 0o666)
                except Exception as e:
                    print(f"Error changing permissions of log file {self.log_file_path}: {e}")  # Ignore chmod errors
            except Exception as e:
                # If file opening fails, log to store but don't fail completely
                self.log_file = None
                # We can't use logging here as it would create recursion, so we'll just continue
                print(f"Error opening log file {self.log_file_path}: {e}")
    
    def emit(self, record):
        try:
            msg = self.format(record)
            with threading.Lock():
                self.log_store.append(msg)
                # Keep only last max_lines to prevent memory issues
                if len(self.log_store) > self.max_lines:
                    self.log_store.pop(0)
            
            # Also write to file if available
            if self.log_file:
                try:
                    with self._file_lock:
                        self.log_file.write(msg + '\n')
                        self.log_file.flush()  # Ensure immediate write
                except Exception:
                    pass  # Ignore file write errors
        except Exception:
            pass  # Ignore errors in logging handler
    
    def close(self):
        """Close the log file if it's open."""
        super().close()
        if self.log_file:
            try:
                with self._file_lock:
                    self.log_file.close()
            except Exception:
                pass
            self.log_file = None

app = FastAPI(title="TIFF Compression Server", version="1.0.0")

# Thread-safe state management
class CompressionState:
    """Thread-safe compression state manager."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._status = "idle"  # idle, running, stopped, completed, error
        self._current_file = None
        self._current_index = 0
        self._total_files = 0
        self._stats = {
            'success_count': 0,
            'skip_count': 0,
            'error_count': 0,
            'consecutive_errors': 0,
            'total_files': 0
        }
        self._start_time = None
        self._error_message = None
        self._results = None
        self._stop_requested = False
        self._logs: List[str] = []
        self._final_elapsed = None
    
    def reset(self):
        """Reset state for a new run."""
        with self._lock:
            self._status = "idle"
            self._current_file = None
            self._current_index = 0
            self._total_files = 0
            self._stats = {
                'success_count': 0,
                'skip_count': 0,
                'error_count': 0,
                'consecutive_errors': 0,
                'total_files': 0
            }
            self._start_time = None
            self._error_message = None
            self._results = None
            self._stop_requested = False
            self._logs = []
            self._final_elapsed = None
    
    def set_running(self):
        """Mark compression as running."""
        with self._lock:
            self._status = "running"
            if self._start_time is None:  # Only set start time if not already set
                self._start_time = time.time()
            self._stop_requested = False
    
    def update_progress(self, current_file: str, current_index: int, total_files: int, stats: Dict):
        """Update progress information."""
        with self._lock:
            self._current_file = current_file
            self._current_index = current_index
            self._total_files = total_files
            self._stats.update(stats)
    
    def set_stopped(self):
        """Mark compression as stopped."""
        with self._lock:
            self._status = "stopped"
            self._stop_requested = True
    
    def set_completed(self, results: Dict):
        """Mark compression as completed."""
        with self._lock:
            self._status = "completed"
            self._results = results
            # Ensure progress shows 100% when completed
            if self._total_files > 0:
                self._current_index = self._total_files
            # Freeze elapsed time at completion
            if self._start_time:
                self._final_elapsed = time.time() - self._start_time
    
    def set_error(self, error_message: str):
        """Mark compression as error."""
        with self._lock:
            self._status = "error"
            self._error_message = error_message
    
    def should_stop(self) -> bool:
        """Check if stop was requested."""
        with self._lock:
            return self._stop_requested
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        with self._lock:
            status = {
                'status': self._status,
                'current_file': self._current_file,
                'current_index': self._current_index,
                'total_files': self._total_files,
                'stats': self._stats.copy(),
                'error_message': self._error_message
            }
            
            if self._start_time:
                # Calculate elapsed time - continue until completed, then freeze
                if self._status == 'running':
                    elapsed = time.time() - self._start_time
                elif self._final_elapsed is not None:
                    # Use frozen elapsed time for completed status
                    elapsed = self._final_elapsed
                else:
                    # For stopped/error, calculate current elapsed
                    elapsed = time.time() - self._start_time
                status['elapsed_seconds'] = elapsed
                if self._total_files > 0 and self._current_index > 0 and self._status == 'running':
                    avg_time_per_file = elapsed / self._current_index
                    remaining_files = self._total_files - self._current_index
                    status['estimated_remaining_seconds'] = avg_time_per_file * remaining_files
            
            if self._results:
                status['results'] = self._results
            
            if self._total_files > 0:
                # Show 100% when completed or stopped
                if self._status in ('completed', 'stopped'):
                    status['progress_percent'] = 100.0
                else:
                    status['progress_percent'] = (self._current_index / self._total_files) * 100
            elif self._status in ('completed', 'stopped'):
                status['progress_percent'] = 100.0
            
            return status
    
    def is_running(self) -> bool:
        """Check if compression is currently running."""
        with self._lock:
            return self._status == "running"
    
    def get_logs(self, since: Optional[int] = None, max_display_lines: int = 1000) -> List[str]:
        """Get logs, optionally starting from a specific index.
        
        Args:
            since: Optional line index to start from
            max_display_lines: Maximum number of lines to return (for display performance)
        """
        with self._lock:
            if since is not None and since < len(self._logs):
                logs = self._logs[since:]
            else:
                logs = self._logs.copy()
            
            # Return only the most recent max_display_lines for performance
            if len(logs) > max_display_lines:
                logs = logs[-max_display_lines:]
            
            return logs
    
    def get_log_count(self) -> int:
        """Get total number of log lines."""
        with self._lock:
            return len(self._logs)
    
    def get_recent_logs(self, max_lines: int = 1000) -> List[str]:
        """Get only the most recent log lines for display."""
        with self._lock:
            if len(self._logs) <= max_lines:
                return self._logs.copy()
            return self._logs[-max_lines:]
    
    def get_log_store(self) -> List[str]:
        """Get reference to log store for handler."""
        return self._logs


# Global state
compression_state = CompressionState()
compression_thread: Optional[threading.Thread] = None
compression_lock = threading.Lock()


# Request/Response models
class CompressionRequest(BaseModel):
    """Request model for starting compression."""
    folder: str = Field(..., description="Folder containing TIFF files to compress")
    compression: str = Field(default="zlib", description="Compression algorithm (zlib or jpeg_2000_lossy)")
    quality: int = Field(default=85, ge=0, le=100, description="Compression quality (0-100)")
    threads: Optional[int] = Field(default=None, description="Number of threads (None for auto)")
    output: Optional[str] = Field(default=None, description="Output folder (None for in-place)")
    dry_run: bool = Field(default=False, description="Dry run mode")
    cleanup_temp: bool = Field(default=False, description="Clean up temporary files")
    cleanup_error_files: bool = Field(default=False, description="Clean up error files")
    verify_lossless_exact: bool = Field(default=False, description="Verify lossless compression exactly")
    ignore_compression_ratio: bool = Field(default=False, description="Ignore compression ratio threshold")
    force_recompress_skipped: bool = Field(default=False, description="Force recompress skipped files")
    force_recompress_processed: bool = Field(default=False, description="Force recompress processed files")
    
    @property
    def preserve_ownership(self) -> bool:
        """Determine preserve_ownership status from environment variable, default field, or provided value."""
        import os
        env_value = os.getenv('PRESERVE_OWNERSHIP', '')
        if env_value.lower() in ('1', 'true', 'yes'):
            return True
        # Fallback to default or provided value if env var not set/enabled
        return getattr(self, 'preserve_ownership_internal', True)

    # The internal field actually stores the input value (but users should access .preserve_ownership property)
    preserve_ownership_internal: bool = Field(default=True, alias="preserve_ownership", description="Preserve file ownership and permissions (useful when running as root in Docker)")


def progress_callback(current_file: str, current_index: int, total_files: int, stats: Dict) -> bool:
    """Progress callback for compression."""
    compression_state.update_progress(current_file, current_index, total_files, stats)
    # Return False if stop was requested, True otherwise
    return not compression_state.should_stop()


def run_compression_thread(request: CompressionRequest):
    """Run compression in a background thread."""
    global compression_state
    
    # Create log file path in the root of the directory being compressed
    log_file_path = os.path.join(
        request.folder,
        f"compression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    # Add log capture handler with file logging
    log_handler = LogCaptureHandler(
        compression_state.get_log_store(),
        log_file_path=log_file_path
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO)
    
    try:
        # State is already set to running in the API endpoint
        # compression_state.set_running()  # Removed - already set in API endpoint
        
        results = run_compression(
            folder=request.folder,
            compression=request.compression,
            quality=request.quality,
            threads=request.threads,
            output=request.output,
            dry_run=request.dry_run,
            cleanup_temp=request.cleanup_temp,
            cleanup_error_files=request.cleanup_error_files,
            verify_lossless_exact=request.verify_lossless_exact,
            ignore_compression_ratio=request.ignore_compression_ratio,
            force_recompress_skipped=request.force_recompress_skipped,
            force_recompress_processed=request.force_recompress_processed,
            progress_callback=progress_callback,
            preserve_ownership=request.preserve_ownership,
            save_log_file=False
        )
        
        if compression_state.should_stop():
            compression_state.set_stopped()
        else:
            compression_state.set_completed(results)
    except Exception as e:
        compression_state.set_error(str(e))
        logging.error(f"Compression error: {e}", exc_info=True)
    finally:
        # Remove log handler and close file
        root_logger.removeHandler(log_handler)
        log_handler.close()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve simple HTML UI."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TIFF Compression Server</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            h1 { color: #333; }
            h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
            label { display: block; margin-top: 10px; font-weight: bold; }
            input, select { width: 100%; padding: 8px; margin-top: 5px; box-sizing: border-box; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px; }
            button:hover { background: #45a049; }
            button.stop { background: #f44336; }
            button.stop:hover { background: #da190b; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            .status { padding: 15px; border-radius: 4px; margin-top: 10px; }
            .status.idle { background: #e3f2fd; }
            .status.running { background: #fff3e0; }
            .status.completed { background: #e8f5e9; }
            .status.stopped { background: #ffebee; }
            .status.error { background: #fce4ec; }
            .progress-bar { width: 100%; height: 30px; background: #ddd; border-radius: 4px; overflow: hidden; margin: 10px 0; }
            .progress-fill { height: 100%; background: #4CAF50; transition: width 0.3s; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 10px; }
            .stat-item { background: white; padding: 10px; border-radius: 4px; }
            .stat-label { font-size: 0.9em; color: #666; }
            .stat-value { font-size: 1.5em; font-weight: bold; color: #333; }
            .log-window { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 4px; margin-top: 10px; font-family: 'Courier New', monospace; font-size: 0.9em; max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; }
            .log-line { margin: 2px 0; }
            .log-line.error { color: #f48771; }
            .log-line.warning { color: #cca700; }
            .log-line.info { color: #4ec9b0; }
        </style>
    </head>
    <body>
        <h1>TIFF Compression Server</h1>
        
        <div class="container">
            <h2>Start Compression</h2>
            <form id="compressForm">
                <label>Folder Path:</label>
                <input type="text" id="folder" name="folder" value="/data" required>
                
                <label>Compression Type:</label>
                <select id="compression" name="compression">
                    <option value="zlib">zlib (lossless)</option>
                    <option value="jpeg_2000_lossy">JPEG 2000 (lossy)</option>
                </select>
                
                <label>Quality (0-100):</label>
                <input type="number" id="quality" name="quality" value="85" min="0" max="100">
                
                <label>Threads (leave empty for auto):</label>
                <input type="number" id="threads" name="threads" min="1">
                
                <label>Output Folder (leave empty for in-place):</label>
                <input type="text" id="output" name="output">
                
                <label>
                    <input type="checkbox" id="dry_run" name="dry_run"> Dry Run
                </label>
                
                <label>
                    <input type="checkbox" id="cleanup_temp" name="cleanup_temp"> Cleanup Temp Files
                </label>
                
                <label>
                    <input type="checkbox" id="verify_lossless_exact" name="verify_lossless_exact"> Verify Lossless Exact
                </label>
                
                <label>
                    <input type="checkbox" id="ignore_compression_ratio" name="ignore_compression_ratio"> Ignore Compression Ratio
                </label>
                
                <label>
                    <input type="checkbox" id="force_recompress_skipped" name="force_recompress_skipped"> Force Recompress Skipped
                </label>
                
                <label>
                    <input type="checkbox" id="force_recompress_processed" name="force_recompress_processed"> Force Recompress Processed
                </label>
                
                <button type="submit" id="startBtn">Start Compression</button>
                <button type="button" id="stopBtn" class="stop" disabled>Stop Compression</button>
            </form>
        </div>
        
        <div class="container">
            <h2>Status</h2>
            <div id="status" class="status idle">
                <div><strong>Status:</strong> <span id="statusText">Idle</span></div>
                <div id="progressContainer" style="display: none;">
                    <div class="progress-bar">
                        <div id="progressFill" class="progress-fill" style="width: 0%"></div>
                    </div>
                    <div>Processing: <span id="currentFile">-</span></div>
                    <div>Progress: <span id="progressText">0 / 0</span></div>
                </div>
                <div id="statsContainer" class="stats" style="display: none;"></div>
                <div id="errorMessage" style="color: red; margin-top: 10px; display: none;"></div>
            </div>
        </div>
        
        <div class="container">
            <h2>Logs</h2>
            <div id="logWindow" class="log-window" style="display: none;"></div>
        </div>
        
        <script>
            const form = document.getElementById('compressForm');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const statusDiv = document.getElementById('status');
            const statusText = document.getElementById('statusText');
            const progressContainer = document.getElementById('progressContainer');
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            const currentFile = document.getElementById('currentFile');
            const statsContainer = document.getElementById('statsContainer');
            const errorMessage = document.getElementById('errorMessage');
            const logWindow = document.getElementById('logWindow');
            
            let statusInterval = null;
            let logInterval = null;
            let lastLogIndex = 0;
            
            async function startCompression() {
                const formData = new FormData(form);
                const data = {
                    folder: formData.get('folder'),
                    compression: formData.get('compression'),
                    quality: parseInt(formData.get('quality')) || 85,
                    threads: formData.get('threads') ? parseInt(formData.get('threads')) : null,
                    output: formData.get('output') || null,
                    dry_run: formData.has('dry_run'),
                    cleanup_temp: formData.has('cleanup_temp'),
                    cleanup_error_files: false,
                    verify_lossless_exact: formData.has('verify_lossless_exact'),
                    ignore_compression_ratio: formData.has('ignore_compression_ratio'),
                    force_recompress_skipped: formData.has('force_recompress_skipped'),
                    force_recompress_processed: formData.has('force_recompress_processed')
                };
                
                try {
                    const response = await fetch('/api/compress', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Failed to start compression');
                    }
                    
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    lastLogIndex = 0;
                    logWindow.style.display = 'block';
                    logWindow.innerHTML = '';
                    updateStatus();
                    updateLogs();
                    statusInterval = setInterval(updateStatus, 1000);
                    logInterval = setInterval(updateLogs, 1000);
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            async function stopCompression() {
                try {
                    const response = await fetch('/api/stop', { method: 'POST' });
                    if (!response.ok) {
                        throw new Error('Failed to stop compression');
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            async function updateStatus() {
                try {
                    const response = await fetch('/api/status');
                    const status = await response.json();
                    
                    statusText.textContent = status.status.toUpperCase();
                    statusDiv.className = 'status ' + status.status;
                    
                    if (status.status === 'running' || status.status === 'completed' || status.status === 'stopped') {
                        progressContainer.style.display = 'block';
                        // Ensure progress shows 100% when completed or stopped
                        let progress = status.progress_percent || 0;
                        if (status.status === 'completed' || status.status === 'stopped') {
                            progress = 100.0;
                        }
                        progressFill.style.width = progress + '%';
                        progressText.textContent = `${status.current_index || status.total_files} / ${status.total_files}`;
                        currentFile.textContent = status.current_file || '-';
                        
                        // Show elapsed time - update while running, freeze when completed
                        if (status.elapsed_seconds !== undefined) {
                            const elapsed = Math.floor(status.elapsed_seconds);
                            const hours = Math.floor(elapsed / 3600);
                            const minutes = Math.floor((elapsed % 3600) / 60);
                            const seconds = elapsed % 60;
                            const timeStr = `${hours}h ${minutes}m ${seconds}s`;
                            if (!document.getElementById('elapsedTime')) {
                                const elapsedDiv = document.createElement('div');
                                elapsedDiv.id = 'elapsedTime';
                                elapsedDiv.className = 'stat-item';
                                statsContainer.appendChild(elapsedDiv);
                            }
                            document.getElementById('elapsedTime').innerHTML = '<div class="stat-label">Elapsed</div><div class="stat-value">' + timeStr + '</div>';
                            statsContainer.style.display = 'grid';
                        } else {
                            statsContainer.style.display = 'none';
                        }
                    } else {
                        progressContainer.style.display = 'none';
                        statsContainer.style.display = 'none';
                    }
                    
                    if (status.status === 'completed' || status.status === 'stopped' || status.status === 'error') {
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                        if (statusInterval) {
                            clearInterval(statusInterval);
                            statusInterval = null;
                        }
                        if (logInterval) {
                            clearInterval(logInterval);
                            logInterval = null;
                        }
                        // Final log update
                        updateLogs();
                    }
                    
                    if (status.status === 'error') {
                        errorMessage.textContent = 'Error: ' + (status.error_message || 'Unknown error');
                        errorMessage.style.display = 'block';
                    } else {
                        errorMessage.style.display = 'none';
                    }
                } catch (error) {
                    console.error('Error updating status:', error);
                }
            }
            
            async function updateLogs() {
                try {
                    // Always fetch all logs (since=0 or no since) to get complete log set
                    // This ensures we have all logs when we clear and re-render
                    const response = await fetch(`/api/logs`);
                    const data = await response.json();
                    
                    if (data.logs && data.logs.length > 0) {
                        // Clear existing logs before adding new ones to prevent duplicates
                        logWindow.innerHTML = '';
                        
                        const fragment = document.createDocumentFragment();
                        data.logs.forEach(line => {
                            const logLine = document.createElement('div');
                            logLine.className = 'log-line';
                            
                            // Color code by log level
                            if (line.includes(' - ERROR - ')) {
                                logLine.classList.add('error');
                            } else if (line.includes(' - WARNING - ')) {
                                logLine.classList.add('warning');
                            } else if (line.includes(' - INFO - ')) {
                                logLine.classList.add('info');
                            }
                            
                            logLine.textContent = line;
                            fragment.appendChild(logLine);
                        });
                        
                        logWindow.appendChild(fragment);
                        lastLogIndex = data.total_lines;
                        
                        // Auto-scroll to bottom
                        logWindow.scrollTop = logWindow.scrollHeight;
                    }
                } catch (error) {
                    console.error('Error updating logs:', error);
                }
            }
            
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                startCompression();
            });
            
            stopBtn.addEventListener('click', stopCompression);
            
            // Initial status update
            updateStatus();
            setInterval(updateStatus, 2000);
        </script>
    </body>
    </html>
    """
    return html


@app.post("/api/compress")
async def start_compression(request: CompressionRequest):
    """Start a compression run."""
    global compression_thread, compression_lock
    
    with compression_lock:
        if compression_state.is_running():
            raise HTTPException(status_code=409, detail="Compression is already running")
        
        # Reset state
        compression_state.reset()
        
        # Validate folder exists
        if not os.path.isdir(request.folder):
            raise HTTPException(status_code=400, detail=f"Folder does not exist: {request.folder}")
        
        # Set running state and start time immediately when API is called
        compression_state.set_running()
        
        # Start compression in background thread
        compression_thread = threading.Thread(
            target=run_compression_thread,
            args=(request,),
            daemon=True
        )
        compression_thread.start()
        
        return {"message": "Compression started", "status": "running"}


@app.get("/api/status")
async def get_status():
    """Get current compression status."""
    return compression_state.get_status()


@app.post("/api/stop")
async def stop_compression():
    """Stop the current compression run."""
    global compression_lock
    
    with compression_lock:
        if not compression_state.is_running():
            raise HTTPException(status_code=409, detail="No compression is currently running")
        
        compression_state.set_stopped()
        return {"message": "Stop requested", "status": "stopping"}


@app.get("/api/logs")
async def get_logs(since: Optional[int] = None):
    """Get compression logs, optionally starting from a specific line index."""
    logs = compression_state.get_logs(since=since)
    return {
        "logs": logs,
        "total_lines": compression_state.get_log_count(),
        "since": since or 0
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

