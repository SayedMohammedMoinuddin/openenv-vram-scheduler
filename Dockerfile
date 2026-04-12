FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first for better caching
COPY --chown=user pyproject.toml requirements.txt uv.lock ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (including the /server folder)
COPY --chown=user . .

# CRITICAL STEP: Install the current project so the "server" command is created
RUN pip install --no-cache-dir .

EXPOSE 7860

# This now works because 'pip install .' created the 'server' executable
CMD ["server"]
