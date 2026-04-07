import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const ENGINE_SCRIPT = '/home/ubuntu/daily_data/scripts/engine.py';
const LOG_DIR = '/home/ubuntu/daily_data/logs';

export async function POST() {
  try {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const logFile = `${LOG_DIR}/engine_manual_${timestamp}.log`;

    // Run engine with --force flag (bypasses trading day check)
    const { stdout, stderr } = await execAsync(
      `python3 -u ${ENGINE_SCRIPT} --force 2>&1 | tee ${logFile}`,
      { timeout: 300000, maxBuffer: 10 * 1024 * 1024 }
    );

    const lines = stdout.split('\n').filter((l: string) => l.trim());
    const lastLines = lines.slice(-10);

    return NextResponse.json({
      success: true,
      message: 'Engine completed',
      log: lastLines,
      timestamp: new Date().toISOString(),
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json(
      { success: false, error: message },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    status: 'ready',
    script: ENGINE_SCRIPT,
    message: 'POST to this endpoint to trigger the engine',
  });
}
