# Build stage
FROM node:20-slim AS build

WORKDIR /app

# Build arguments for API URL and STT mode
ARG REACT_APP_API_URL=http://localhost:3000
ARG REACT_APP_STT_MODE=api
ENV REACT_APP_API_URL=$REACT_APP_API_URL
ENV REACT_APP_STT_MODE=$REACT_APP_STT_MODE

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage - serve with nginx
FROM nginx:alpine

# Copy built assets
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx config for SPA routing
RUN echo 'server { \
    listen 80; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html; \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
