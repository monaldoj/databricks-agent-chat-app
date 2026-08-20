import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';

const greetingImageSchema = z.object({
  /** Filename relative to the branding directory. */
  file: z.string().min(1),
  alt: z.string().optional(),
  /** CSS max-width (e.g. "22rem"). */
  maxWidth: z.string().optional(),
  /** CSS max-height (e.g. "6rem"). */
  maxHeight: z.string().optional(),
});

/**
 * Local UI overrides loaded from `ui-branding/config.json`.
 * Unknown keys are ignored so the file can grow without a server change
 * for unused fields; add them here when the UI should honor them.
 */
export const uiBrandingSchema = z.object({
  greeting: z.string().optional(),
  greetingImage: greetingImageSchema.optional(),
});

export type UiBrandingConfig = z.infer<typeof uiBrandingSchema>;

export type LoadedUiBranding = {
  dir: string;
  config: UiBrandingConfig;
};

const moduleDir = path.dirname(fileURLToPath(import.meta.url));

function brandingDisabled(): boolean {
  return process.env.PLAYWRIGHT === 'True';
}

function candidateDirs(): string[] {
  const envDir = process.env.UI_BRANDING_DIR?.trim();
  if (envDir) {
    return [path.resolve(envDir)];
  }

  return [
    path.resolve(process.cwd(), 'ui-branding'),
    path.resolve(process.cwd(), '../ui-branding'),
    // Bundled server: e2e-chatbot-app-next/server/dist
    path.resolve(moduleDir, '../../../ui-branding'),
    // tsx: e2e-chatbot-app-next/server/src/lib
    path.resolve(moduleDir, '../../../../ui-branding'),
    path.resolve(moduleDir, '../../ui-branding'),
  ];
}

export function resolveBrandingDir(): string | null {
  if (brandingDisabled()) {
    return null;
  }

  for (const dir of candidateDirs()) {
    if (fs.existsSync(path.join(dir, 'config.json'))) {
      return dir;
    }
  }
  return null;
}

export function loadUiBranding(): LoadedUiBranding | null {
  const dir = resolveBrandingDir();
  if (!dir) {
    return null;
  }

  const configPath = path.join(dir, 'config.json');
  try {
    const raw = fs.readFileSync(configPath, 'utf8');
    const parsedJson: unknown = JSON.parse(raw);
    const parsed = uiBrandingSchema.safeParse(parsedJson);
    if (!parsed.success) {
      console.warn(
        `[ui-branding] Invalid ${configPath}: ${parsed.error.message}`,
      );
      return null;
    }
    if (parsedJson && typeof parsedJson === 'object') {
      const extra = Object.keys(parsedJson).filter(
        (key) => key !== 'greeting' && key !== 'greetingImage',
      );
      if (extra.length > 0) {
        console.warn(
          `[ui-branding] Unused keys in ${configPath}: ${extra.join(', ')}`,
        );
      }
    }
    return { dir, config: parsed.data };
  } catch (error) {
    console.warn(
      `[ui-branding] Could not read ${configPath}:`,
      error instanceof Error ? error.message : error,
    );
    return null;
  }
}

export function resolveBrandingAsset(
  dir: string,
  relativePath: string,
): string | null {
  if (!relativePath || relativePath.includes('\0')) {
    return null;
  }

  const resolved = path.resolve(dir, relativePath);
  const relative = path.relative(dir, resolved);
  if (
    relative.startsWith('..') ||
    path.isAbsolute(relative) ||
    relative === ''
  ) {
    return null;
  }

  try {
    if (fs.statSync(resolved).isFile()) {
      return resolved;
    }
  } catch {
    return null;
  }
  return null;
}

export function brandingAssetUrl(file: string): string {
  const encoded = file
    .split('/')
    .filter((segment) => segment.length > 0)
    .map((segment) => encodeURIComponent(segment))
    .join('/');
  return `/api/ui-branding/${encoded}`;
}
