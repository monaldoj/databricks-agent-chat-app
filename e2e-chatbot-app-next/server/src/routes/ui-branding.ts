import {
  Router,
  type Request,
  type Response,
  type Router as RouterType,
} from 'express';
import { resolveBrandingAsset, resolveBrandingDir } from '../lib/ui-branding';

export const uiBrandingRouter: RouterType = Router();

/**
 * GET /api/ui-branding/* — serve files from the local ui-branding directory.
 */
uiBrandingRouter.use((req: Request, res: Response) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.status(405).end();
    return;
  }

  const dir = resolveBrandingDir();
  let relativePath: string;
  try {
    relativePath = decodeURIComponent(req.path.replace(/^\//, ''));
  } catch {
    res.status(400).end();
    return;
  }
  if (!dir || !relativePath) {
    res.status(404).end();
    return;
  }

  const asset = resolveBrandingAsset(dir, relativePath);
  if (!asset) {
    res.status(404).end();
    return;
  }

  res.sendFile(asset);
});
