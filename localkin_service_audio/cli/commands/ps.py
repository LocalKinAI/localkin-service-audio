"""
PS command - Show running LocalKin Audio servers.
"""
import socket

import click
import requests

from ..utils import print_header, print_success, print_info


COMMON_PORTS = [8000, 8001, 8002, 8003, 8004, 8005]


def _check_port(port: int) -> bool:
    """Check if a port is open on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("localhost", port)) == 0


def _check_localkin_server(port: int) -> dict | None:
    """Check if a LocalKin Audio server is running on the given port."""
    try:
        resp = requests.get(f"http://localhost:{port}/", timeout=2)
        if resp.status_code != 200 or "LocalKin" not in resp.text:
            return None
    except Exception:
        return None

    model_name = "Unknown"
    model_type = "Unknown"
    try:
        models_resp = requests.get(f"http://localhost:{port}/models", timeout=2)
        if models_resp.status_code == 200:
            data = models_resp.json()
            model_name = data.get("model_name", "Unknown")
            model_type = data.get("model_type", "Unknown")
    except Exception:
        pass

    return {
        "port": port,
        "model": model_name,
        "type": model_type,
        "url": f"http://localhost:{port}",
    }


def _get_health_status(port: int) -> str:
    """Get the health status emoji for a server."""
    try:
        resp = requests.get(f"http://localhost:{port}/health", timeout=2)
        return "Running" if resp.status_code == 200 else "Issues"
    except Exception:
        return "Offline"


@click.command()
def ps():
    """
    Show running LocalKin Audio servers.

    Scans common ports (8000-8005) for active servers.

    Examples:

        kin audio ps
    """
    print_header("Running Servers")
    print_info("Scanning for LocalKin Audio servers...")

    running_servers = []
    for port in COMMON_PORTS:
        try:
            if _check_port(port):
                server = _check_localkin_server(port)
                if server:
                    running_servers.append(server)
        except Exception:
            continue

    if not running_servers:
        print_info("No running LocalKin Audio servers found.")
        print(f"\n  Start a server with: kin audio serve <model_name> --port <port>")
        return

    print_success(f"Found {len(running_servers)} running server(s)\n")
    print(f"  {'PORT':<8} {'MODEL':<20} {'TYPE':<8} {'URL':<30} {'STATUS'}")
    print("  " + "-" * 76)

    for server in running_servers:
        status = _get_health_status(server["port"])
        print(
            f"  {server['port']:<8} {server['model']:<20} {server['type']:<8} "
            f"{server['url']:<30} {status}"
        )

    print(f"\n  Access API docs at http://localhost:<PORT>/docs")
