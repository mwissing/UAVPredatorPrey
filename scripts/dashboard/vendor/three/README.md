# Three.js 0.186.1 (vendored)

WebGL library for the dashboard's 3D replay (`../../replay.js`). Kept in the repository, pinned, so the
dashboard works offline and renders the same way later. MIT license, see `LICENSE`.

Source: <https://cdn.jsdelivr.net/npm/three@0.186.1/>

| File | From |
|---|---|
| `three.module.min.js` | `build/three.module.min.js` (jsDelivr's minified `build/three.module.js`) |
| `three.core.js` | `build/three.core.min.js`, saved under the name `three.module.min.js` imports |
| `addons/controls/OrbitControls.js` | `examples/jsm/controls/OrbitControls.js` |
| `addons/lines/*.js` | `examples/jsm/lines/` (thick lines for the trails) |

The page maps `three` and `three/addons/` to these files with an import map. `dashboard.py build` copies
this folder to `artifacts/dashboard/vendor/three/`, where the server serves it.
