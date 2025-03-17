"""
Gunicorn configuration file for optimizing performance on Render
"""

import os
import multiprocessing

# Bind to the port provided by Render
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Use minimal workers for Render free tier
workers = 1

# Use a single thread per worker to reduce memory usage
threads = 1

# Timeout for worker processes (in seconds)
timeout = 120

# Maximum number of requests a worker will process before restarting
max_requests = 1000
max_requests_jitter = 50

# Log level
loglevel = 'info'

# Preload the application to save memory
preload_app = True

# Worker class
worker_class = 'sync'

# Limit the maximum number of simultaneous clients
worker_connections = 100

# Restart workers when code changes (development only)
reload = os.environ.get('FLASK_ENV') == 'development'

# Graceful timeout
graceful_timeout = 30

# Keep-alive timeout
keepalive = 5
