#!/bin/sh
set -e

if [ -n "$SSL_CERT" ] && [ -n "$SSL_KEY" ] && [ -s "$SSL_CERT" ] && [ -s "$SSL_KEY" ]; then
  echo "[nginx] HTTPS enabled"
  cat > /etc/nginx/conf.d/default.conf << EOF
server {
    listen 80;
    listen 443 ssl;
    ssl_certificate ${SSL_CERT};
    ssl_certificate_key ${SSL_KEY};
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }
}
EOF
else
  echo "[nginx] HTTP mode (no SSL certs)"
fi

exec nginx -g "daemon off;"
