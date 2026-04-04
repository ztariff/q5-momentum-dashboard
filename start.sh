#!/bin/sh
echo "Starting on port: ${PORT:-3000}"
exec npx next start -H 0.0.0.0 -p "${PORT:-3000}"
