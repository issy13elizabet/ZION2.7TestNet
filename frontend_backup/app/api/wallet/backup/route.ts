import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function POST(request: Request) {
  try {
    const { type, recipient } = await request.json();
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    switch (type) {
      case 'basic':
        // Execute basic backup script
        try {
          const { stdout } = await execAsync('cd ../../../scripts && ./backup-wallet.sh');
          return NextResponse.json({
            success: true,
            message: 'Basic wallet backup completed!',
            output: stdout,
            timestamp
          });
        } catch (error) {
          return NextResponse.json({
            success: false,
            message: 'Backup script execution failed',
            error: (error as Error).message
          }, { status: 500 });
        }

      case 'gpg':
        if (!recipient) {
          return NextResponse.json({ error: 'GPG recipient required' }, { status: 400 });
        }
        
        // Execute secure GPG backup script
        try {
          const { stdout } = await execAsync(`cd ../../../scripts && ./secure-backup.sh zion-walletd "${recipient}"`);
          return NextResponse.json({
            success: true,
            message: 'GPG encrypted backup completed!',
            output: stdout,
            recipient,
            timestamp
          });
        } catch (error) {
          return NextResponse.json({
            success: false,
            message: 'GPG backup failed',
            error: (error as Error).message
          }, { status: 500 });
        }

      default:
        return NextResponse.json({ error: 'Invalid backup type' }, { status: 400 });
    }

  } catch (error) {
    console.error('Backup error:', error);
    return NextResponse.json(
      { error: 'Backup operation failed' },
      { status: 500 }
    );
  }
}