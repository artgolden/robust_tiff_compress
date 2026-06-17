# Building and launching a docker image
_Exported on 16/06/2026 at 16:26:59 CEST from Cursor (3.0.12)_

---

**User**

Guide on how to build the docker image for this project and launch it

---

**Cursor**

Reviewing the Dockerfile and project structure to prepare a build and launch guide.


Checking the webserver structure to understand the application:


## Building and Running the Docker Image

### Prerequisites
- Docker installed and running
- Access to the project directory

### Step 1: Build the Docker Image

From the project root (`/home/tema/work/src/light_sheet_processing/robust_tiff_compress`), run:

```bash
docker build -t robust-tiff-compress .
```

This will:
- Use Python 3.12-slim as the base image
- Install system dependencies (build tools, image libraries)
- Install Python dependencies from `requirements.txt`
- Copy your application code
- Set up the webserver to run on port 8000

**Optional: Tag with a version**
```bash
docker build -t robust-tiff-compress:latest .
# or
docker build -t robust-tiff-compress:v1.0.0 .
```

### Step 2: Run the Docker Container

**Basic run:**
```bash
docker run -d -p 8000:8000 --name tiff-compress-server robust-tiff-compress
```

**With data volume mount (recommended):**
If you need to access files from your host machine:
```bash
docker run -d -p 8000:8000 \
  -v /path/to/your/data:/data \
  --name tiff-compress-server \
  robust-tiff-compress
```

**With custom port:**
If you want to use a different host port (e.g., 8080):
```bash
docker run -d -p 8080:8000 --name tiff-compress-server robust-tiff-compress
```

**With custom container port:**
To change the container's internal port:
```bash
docker run -d -p 8000:9000 \
  -e PORT=9000 \
  --name tiff-compress-server \
  robust-tiff-compress
```

### Step 3: Verify the Container is Running

```bash
# Check if container is running
docker ps

# View logs
docker logs tiff-compress-server

# Follow logs in real-time
docker logs -f tiff-compress-server
```

### Step 4: Access the Web Interface

Once running, access the FastAPI web interface at:
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

### Useful Docker Commands

**Stop the container:**
```bash
docker stop tiff-compress-server
```

**Start a stopped container:**
```bash
docker start tiff-compress-server
```

**Remove the container:**
```bash
docker rm tiff-compress-server
```

**Remove the image:**
```bash
docker rmi robust-tiff-compress
```

**Run interactively (for debugging):**
```bash
docker run -it --rm -p 8000:8000 robust-tiff-compress /bin/bash
```

### Notes

1. The `.dockerignore` file excludes test files, cache files, and other unnecessary files from the build context.
2. The container exposes port 8000 by default, but you can override it with the `PORT` environment variable.
3. The `/data` directory is created in the container for mounting external data volumes.
4. The webserver runs using `uvicorn` and listens on `0.0.0.0` to accept connections from outside the container.

### Troubleshooting

If you encounter issues:
- Check logs: `docker logs tiff-compress-server`
- Verify the port isn't already in use: `netstat -tuln | grep 8000` or `lsof -i :8000`
- Ensure Docker has enough resources allocated
- Rebuild if you've made code changes: `docker build --no-cache -t robust-tiff-compress .`

```shellscript
docker build -t robust-tiff-compress .
```

```shellscript
docker build -t robust-tiff-compress:latest .
# or
docker build -t robust-tiff-compress:v1.0.0 .
```

```shellscript
docker run -d -p 8000:8000 --name tiff-compress-server robust-tiff-compress
```

```shellscript
docker run -d -p 8000:8000 \
  -v /path/to/your/data:/data \
  --name tiff-compress-server \
  robust-tiff-compress
```

```shellscript
docker run -d -p 8080:8000 --name tiff-compress-server robust-tiff-compress
```

```shellscript
docker run -d -p 8000:9000 \
  -e PORT=9000 \
  --name tiff-compress-server \
  robust-tiff-compress
```

```shellscript
# Check if container is running
docker ps

# View logs
docker logs tiff-compress-server

# Follow logs in real-time
docker logs -f tiff-compress-server
```

```shellscript
docker stop tiff-compress-server
```

```shellscript
docker start tiff-compress-server
```

```shellscript
docker rm tiff-compress-server
```

```shellscript
docker rmi robust-tiff-compress
```

```shellscript
docker run -it --rm -p 8000:8000 robust-tiff-compress /bin/bash
```

