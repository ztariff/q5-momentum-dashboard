import { NextResponse } from 'next/server';
import http from 'http';

function callEngine(): Promise<{ success: boolean; log?: string[]; error?: string }> {
  return new Promise((resolve) => {
    const req = http.request(
      { hostname: '127.0.0.1', port: 3001, path: '/run', method: 'POST', timeout: 300000 },
      (res) => {
        let data = '';
        res.on('data', (chunk: string) => { data += chunk; });
        res.on('end', () => {
          try {
            resolve(JSON.parse(data));
          } catch {
            resolve({ success: false, error: 'Invalid response from engine' });
          }
        });
      }
    );
    req.on('error', (err: Error) => {
      resolve({ success: false, error: `Engine helper not reachable: ${err.message}` });
    });
    req.on('timeout', () => {
      req.destroy();
      resolve({ success: false, error: 'Engine timed out (5 min)' });
    });
    req.end();
  });
}

export async function POST() {
  const result = await callEngine();
  const status = result.success ? 200 : 500;
  return NextResponse.json(result, { status });
}

export async function GET() {
  return NextResponse.json({ status: 'ready', message: 'POST to trigger engine' });
}
