import json
import os
import subprocess
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def write_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}-", delete=False
        ) as output:
            temporary = output.name
            json.dump(value, output, indent=2, ensure_ascii=False)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def command(arguments, log, *, capture=False, timeout=1800):
    """Run argument arrays, saving stdout/stderr and preserving useful failures."""
    with log.open("a", encoding="utf-8") as output:
        output.write("\nCOMMAND " + json.dumps([str(arg) for arg in arguments]) + "\n")
        output.flush()
        result = subprocess.run(
            [str(arg) for arg in arguments],
            stdout=subprocess.PIPE if capture else output,
            stderr=output,
            text=True,
            timeout=timeout,
            check=False,
        )
        if capture:
            output.write(result.stdout)
    if result.returncode:
        with log.open("rb") as output:
            output.seek(max(0, log.stat().st_size - 16384))
            tail = "\n".join(output.read().decode("utf-8", errors="replace").splitlines()[-20:])
        raise RuntimeError(f"Command exited {result.returncode}; see {log}\n{tail}")
    return result.stdout if capture else ""


def request_json(base_url, route, payload=None, *, timeout=180):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        base_url + route,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
            return response.status, json.loads(data) if data else None
    except HTTPError as error:
        data = error.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(data)
        except ValueError:
            pass
        return error.code, data


def require(condition, message):
    # Explicit checks are active even when Python is invoked with -O.
    if not condition:
        raise AssertionError(message)
