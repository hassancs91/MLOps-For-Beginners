#!/usr/bin/env python3
"""
Container Health Check Script for FastAPI Application

This script performs a health check on the FastAPI application running inside
the Docker container. It's used by Docker's HEALTHCHECK instruction to determine
if the container is healthy and ready to serve requests.

The script:
1. Sends an HTTP GET request to the /health endpoint
2. Verifies the response status and content
3. Returns appropriate exit codes for Docker

Exit Codes:
- 0: Container is healthy (success)
- 1: Container is unhealthy (failure)

Usage:
    python healthcheck.py
"""

import sys
import json
import urllib.request
import urllib.error
from typing import Dict, Any


def check_health(host: str = "localhost", port: int = 8000, timeout: int = 5) -> bool:
    """
    Check the health of the FastAPI application.
    
    This function sends a GET request to the /health endpoint and validates
    the response to ensure the API server and model are properly loaded.
    
    Args:
        host: The hostname where the API is running (default: localhost)
        port: The port number where the API is listening (default: 8000)
        timeout: Request timeout in seconds (default: 5)
    
    Returns:
        bool: True if the service is healthy, False otherwise
    """
    
    # Construct the health check URL
    health_url = f"http://{host}:{port}/health"
    
    try:
        # Create the request with a timeout to prevent hanging
        request = urllib.request.Request(health_url)
        
        # Send the GET request to the health endpoint
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # Check if we got a successful HTTP status code (200)
            if response.status != 200:
                print(f"Health check failed: HTTP {response.status}")
                return False
            
            # Parse the JSON response
            data = json.loads(response.read().decode('utf-8'))
            
            # Validate the response structure and content
            # The health endpoint should return status="healthy" and model_loaded=True
            if data.get('status') == 'healthy' and data.get('model_loaded') == True:
                print(f"Health check passed: {data}")
                return True
            else:
                print(f"Health check failed: Service unhealthy - {data}")
                return False
                
    except urllib.error.URLError as e:
        # Handle connection errors (service not running or not reachable)
        print(f"Health check failed: Cannot connect to {health_url} - {e}")
        return False
        
    except json.JSONDecodeError as e:
        # Handle invalid JSON responses
        print(f"Health check failed: Invalid JSON response - {e}")
        return False
        
    except Exception as e:
        # Handle any other unexpected errors
        print(f"Health check failed: Unexpected error - {e}")
        return False


def main() -> int:
    """
    Main function that runs the health check and returns the appropriate exit code.
    
    This is the entry point when the script is run directly. It performs the health
    check and exits with the appropriate code for Docker to interpret.
    
    Returns:
        int: Exit code (0 for healthy, 1 for unhealthy)
    """
    
    # Perform the health check
    is_healthy = check_health()
    
    # Return appropriate exit code
    # Docker interprets exit code 0 as healthy, any non-zero as unhealthy
    if is_healthy:
        sys.exit(0)  # Success - container is healthy
    else:
        sys.exit(1)  # Failure - container is unhealthy


if __name__ == "__main__":
    # Run the health check when script is executed directly
    main()