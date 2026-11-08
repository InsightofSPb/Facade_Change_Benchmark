// Browser-independent mask and coordinate operations. Inputs use the exported native grid.
export function decodeMask(rgba, width, height) {
  if (rgba.length !== width * height * 4) throw new Error('Размер маски не соответствует изображению');
  const packed = new Uint16Array(width * height), support = new Uint8Array(width * height);
  for (let i = 0; i < packed.length; i++) { packed[i] = rgba[i * 4] + 256 * rgba[i * 4 + 1]; support[i] = rgba[i * 4 + 2] === 255 ? 1 : 0; }
  return { packed, support, width, height };
}
export function compareMasks(a, b, selectedBits = 65535) {
  if (a.width !== b.width || a.height !== b.height) throw new Error('Маски находятся на разных сетках');
  const result = new Uint8Array(a.packed.length); let valid = 0, changed = 0;
  for (let i = 0; i < result.length; i++) {
    if (!(a.support[i] && b.support[i] && a.packed[i] && b.packed[i])) continue;
    valid++;
    const added = ((~a.packed[i]) & b.packed[i] & selectedBits) !== 0;
    const removed = (a.packed[i] & (~b.packed[i]) & selectedBits) !== 0;
    result[i] = added && removed ? 3 : added ? 1 : removed ? 2 : 0;
    if (result[i]) changed++;
  }
  return { values: result, valid, changed };
}
export function cropPlacement(facade, mode) {
  const [x, y, w, h] = facade.crop_xywh;
  return mode === 'crop' ? {x:0,y:0,w,h,width:w,height:h} : {x,y,w,h,width:facade.reference_width,height:facade.reference_height};
}
export function cropSliderCut(slider, placement) {
  return Math.max(0, Math.min(100, ((slider / 100) * placement.width - placement.x) / placement.w * 100));
}
export function pointToCrop(px, py, placement) {
  const x = Math.floor(px * placement.width - placement.x), y = Math.floor(py * placement.height - placement.y);
  return x >= 0 && y >= 0 && x < placement.w && y < placement.h ? [x,y] : null;
}
export function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
export function safeWebUrl(value) { try {const u = new URL(value); return ['http:', 'https:'].includes(u.protocol) ? u.href : null;} catch {return null;} }
