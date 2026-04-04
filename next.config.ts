import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  env: {
    POLYGON_API_KEY: process.env.POLYGON_API_KEY || 'cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF',
  },
};

export default nextConfig;
