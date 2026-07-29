/**
 * Resolve a file in `public/data/` against the app's deploy base.
 *
 * An absolute `/data/...` path breaks on GitHub Pages project sites, which
 * serve the app from `/<repo>/` rather than the domain root. `BASE_URL` is
 * whatever `base` is set to in vite.config.ts (`'./'` here) and always ends in
 * a slash, so it can be concatenated directly.
 */
export function dataUrl(filename: string): string {
  return `${import.meta.env.BASE_URL}data/${filename}`
}
