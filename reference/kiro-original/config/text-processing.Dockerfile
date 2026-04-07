FROM node:22-alpine

WORKDIR /app

# Install the npm package that has text processing
RUN npm install -g @modelcontextprotocol/server-everything

# Create entrypoint
ENTRYPOINT ["npx", "@modelcontextprotocol/server-everything"]
