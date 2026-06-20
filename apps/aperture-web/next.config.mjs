/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The data layer is mock-first today; the BFF base URL is wired via env when live.
  env: {
    APERTURE_BFF_URL: process.env.APERTURE_BFF_URL ?? '',
  },
};

export default nextConfig;
