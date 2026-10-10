// 3D replay of evaluated episodes: the arena, its pillars and the four drones with fading trails.
//
// The dashboard page imports this module when the Replay section comes near the viewport. dashboard.py writes, per
// evaluation, replay/<id>-v<format>/index.json (one line per episode: length, how it ended, whether a predator went
// down, plus a few highlights) and <number>.json per episode: the x, y, z of the prey and of each predator at every
// policy step (50 Hz), the pillar positions and the frame at which each predator went down (crash or out of bounds).
// The server extracts an episode the first time the page asks for it. Coordinates are the simulator's: metres,
// z up, arena centre at (0, 0, 6).
//
// Browsing: a category lists its episodes in a fixed order; ◀ ▶ and Random step through it. ★ keeps an episode in
// a starred list stored in this browser, and the address bar always links to the episode shown
// (#replay/<source>/<number>), so a link or bookmark opens it again.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);  // the simulator's z axis points up

const BACKGROUND = new THREE.Color(0x0e1014);
const ROLE = { predator: new THREE.Color(0xeb6834), prey: new THREE.Color(0x3987e5), down: new THREE.Color(0x6f7175) };
const TRAIL_S = 2;         // seconds of trail behind each drone
const DRONE_R = 0.14;      // m; drawn larger than the real drones so they stay visible from afar
const HOLD_S = 1.5;        // the last frame stays this long before the episode starts again
const FLASH_S = 0.7;       // duration of the flash at a catch or crash (simulated seconds)
const SPEEDS = [0.25, 0.5, 1, 2];
const H_FOV = 60;          // degrees: the horizontal field of view of the camera that records the videos
const VIEWS = {            // "video" is camera D of the recorded videos
  video: { label: "Video view", eye: [10, -10, 12.5], target: [0, 0, 5] },
  top: { label: "Top", eye: [0, -0.01, 22], target: [0, 0, 6] },
  side: { label: "Side", eye: [0, -17, 6], target: [0, 0, 6] },
  follow: { label: "Follow prey" },
};
// Categories of the episode index; each lists its episodes in an order that puts the interesting ones first.
const CATEGORIES = [
  { key: "highlights", label: "Highlights", one: "Highlight", order: "" },
  { key: "catch", label: "Catches", one: "Catch", order: "longest first" },
  { key: "prey-crash", label: "Prey hits a pillar", one: "Prey crash", order: "longest first" },
  { key: "down", label: "Predator down", one: "Predator down", order: "longest first" },
  { key: "timeout", label: "Escapes", one: "Escape", order: "nearest misses first" },
  { key: "prey-oob", label: "Prey left the arena", one: "Prey out", order: "longest first" },
  { key: "all", label: "All", one: "Episode", order: "by number" },
];
const STARS_KEY = "uav-replay-stars";

const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

function buildLists(ix) {
  const ids = Array.from({ length: ix.episodes_total }, (_, i) => i);
  const longestFirst = list => list.sort((a, b) => ix.length[b] - ix.length[a] || a - b);
  const closest = i => (ix.closest ? ix.closest[i] : 0);
  return {
    highlights: ix.highlights.map(h => h.index),
    catch: longestFirst(ids.filter(i => ix.end[i] === "catch")),
    "prey-crash": longestFirst(ids.filter(i => ix.end[i] === "prey-crash")),
    down: longestFirst(ids.filter(i => ix.down[i])),
    timeout: ids.filter(i => ix.end[i] === "timeout").sort((a, b) => closest(a) - closest(b) || a - b),
    "prey-oob": longestFirst(ids.filter(i => ix.end[i] === "prey-oob")),
    all: ids,
  };
}

export function mount(root, sources) {
  // ---------------------------------------------------------------- controls and overlays
  root.replaceChildren();
  const head = el("div", "replay-head");
  const sourceSel = el("select");
  sourceSel.setAttribute("aria-label", "Evaluation");
  for (const s of sources) { const o = el("option", "", s.title); o.value = s.id; sourceSel.appendChild(o); }
  const cats = el("div", "replay-chips");
  cats.setAttribute("role", "group");
  cats.setAttribute("aria-label", "Category");
  head.append(sourceSel, cats);

  const nav = el("div", "replay-nav");
  const prevBtn = el("button", "rbtn", "◀"), nextBtn = el("button", "rbtn", "▶");
  prevBtn.setAttribute("aria-label", "Previous episode");
  nextBtn.setAttribute("aria-label", "Next episode");
  const where = el("span", "replay-where");
  const randomBtn = el("button", "rbtn", "Random");
  const starBtn = el("button", "rbtn star", "☆ Star");
  const numLabel = el("label", "replay-num", "Episode");
  const numInput = el("input");
  numInput.type = "number"; numInput.min = "1"; numInput.inputMode = "numeric";
  numLabel.appendChild(numInput);
  nav.append(prevBtn, where, nextBtn, randomBtn, starBtn, numLabel);
  const starsRow = el("div", "replay-stars");
  const note = el("p", "replay-note");

  const stage = el("div", "replay-stage");
  const canvas = el("canvas");
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "img");
  const info = el("div", "replay-hud info"), clock = el("div", "replay-hud clock");
  const legend = el("div", "replay-legend");
  for (const [name, color] of [["predators", ROLE.predator], ["prey", ROLE.prey], ["down", ROLE.down]]) {
    const key = el("span", "", name);
    key.style.setProperty("--c", `#${color.getHexString()}`);
    legend.appendChild(key);
  }
  const msg = el("div", "replay-msg", "Loading…");
  stage.append(canvas, info, clock, legend, msg);

  const bar = el("div", "replay-bar");
  const playBtn = el("button", "rbtn play");
  const slider = el("input");
  slider.type = "range"; slider.min = "0"; slider.step = "1";
  slider.setAttribute("aria-label", "Time");
  const speedSel = el("select");
  speedSel.setAttribute("aria-label", "Playback speed");
  for (const s of SPEEDS) { const o = el("option", "", `${s}×`); o.value = String(s); speedSel.appendChild(o); }
  speedSel.value = "1";
  const pathBtn = el("button", "rbtn", "Full paths");
  pathBtn.setAttribute("aria-pressed", "false");
  const views = el("div", "replay-views");
  for (const [key, v] of Object.entries(VIEWS)) { const b = el("button", "rbtn", v.label); b.dataset.view = key; views.appendChild(b); }
  bar.append(playBtn, slider, speedSel, pathBtn, views);
  for (const b of [prevBtn, nextBtn, randomBtn, starBtn, playBtn, pathBtn, ...views.children]) b.type = "button";
  root.append(head, nav, starsRow, note, stage, bar);

  // ---------------------------------------------------------------- scene
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  } catch (err) {
    msg.textContent = "The 3D view needs WebGL, which this browser does not provide.";
    return;
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene();
  scene.background = BACKGROUND;
  scene.fog = new THREE.Fog(BACKGROUND, 24, 52);
  const camera = new THREE.PerspectiveCamera(36, 16 / 9, 0.1, 200);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.minDistance = 2;
  controls.maxDistance = 45;
  scene.add(new THREE.HemisphereLight(0xdde6f5, 0x202226, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.8);
  sun.position.set(6, -9, 18);
  scene.add(sun);
  const grid = new THREE.GridHelper(24, 24, 0x2c313a, 0x22262d);
  grid.rotation.x = Math.PI / 2;  // GridHelper lies in the x-z plane; the floor here is x-y
  scene.add(grid);
  const arena = new THREE.Group(), pillars = new THREE.Group(), paths = new THREE.Group(), flashes = new THREE.Group();
  scene.add(arena, pillars, paths, flashes);

  const lineMaterials = new Set();  // LineMaterial needs the canvas size to draw widths in pixels
  const lineMaterial = (linewidth, opacity = 1) => {
    const m = new LineMaterial({ linewidth, vertexColors: true, fog: true, transparent: opacity < 1, opacity });
    m.resolution.set(stage.clientWidth || 1, stage.clientHeight || 1);
    lineMaterials.add(m);
    return m;
  };
  const droneGeo = new THREE.SphereGeometry(DRONE_R, 24, 16);
  const flashGeo = new THREE.SphereGeometry(1, 24, 16);
  let pillarGeo = null;
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0xc9ccd2, roughness: 0.55, metalness: 0.1 });

  function buildArena(ix) {
    if (arena.children.length) return;  // the same arena in every evaluation
    const [cx, cy, cz] = ix.arena.center, r = ix.arena.radius;
    arena.position.set(cx, cy, cz);
    arena.add(new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32),
      new THREE.MeshBasicMaterial({ color: 0x5d6b80, transparent: true, opacity: 0.05, depthWrite: false })));
    const ringMat = new THREE.LineBasicMaterial({ color: 0x5d6b80, transparent: true, opacity: 0.4 });
    const circle = [];
    for (let i = 0; i <= 96; i++) { const a = (i / 96) * Math.PI * 2; circle.push(new THREE.Vector3(Math.cos(a) * r, Math.sin(a) * r, 0)); }
    const ringGeo = new THREE.BufferGeometry().setFromPoints(circle);
    const equator = new THREE.Line(ringGeo, ringMat), meridianX = new THREE.Line(ringGeo, ringMat), meridianY = new THREE.Line(ringGeo, ringMat);
    meridianX.rotation.x = Math.PI / 2;
    meridianY.rotation.y = Math.PI / 2;
    arena.add(equator, meridianX, meridianY);
    pillarGeo = new THREE.CylinderGeometry(ix.pillar.radius, ix.pillar.radius, ix.pillar.height, 32);
  }

  // ---------------------------------------------------------------- state
  const indexCache = new Map(), episodeCache = new Map();
  let source = null, ix = null, lists = null, category = "highlights", position = 0;
  let sourceRequest = 0, episodeRequest = 0;
  let ep = null, drones = [], events = [];
  let frame = 0, playing = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  playBtn.textContent = playing ? "Pause" : "Play";
  let speed = 1, follow = false, holdUntil = 0, shownDowns = "";
  let visible = false, raf = 0, ticking = false, last = 0;
  const tmp = new THREE.Vector3(), tmpColor = new THREE.Color();

  const fetchJson = url => fetch(url).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); });
  const getIndex = s => {
    if (!indexCache.has(s.id)) indexCache.set(s.id, fetchJson(`${s.base}index.json`).catch(err => { indexCache.delete(s.id); throw err; }));
    return indexCache.get(s.id);
  };
  const getEpisode = (s, i) => {  // the server extracts an episode the first time it is asked for
    const key = `${s.id}/${i}`;
    if (!episodeCache.has(key)) episodeCache.set(key, fetchJson(`${s.base}${i + 1}.json`).catch(err => { episodeCache.delete(key); throw err; }));
    return episodeCache.get(key);
  };

  // Position at a fractional frame, interpolated between the 50 Hz samples.
  const at = (track, f, out) => {
    const n = track.length / 3 - 1;
    const f0 = Math.max(0, Math.min(Math.floor(f), n)), f1 = Math.min(f0 + 1, n), a = Math.min(Math.max(f - f0, 0), 1);
    return out.set(
      track[3 * f0] + (track[3 * f1] - track[3 * f0]) * a,
      track[3 * f0 + 1] + (track[3 * f1 + 1] - track[3 * f0 + 1]) * a,
      track[3 * f0 + 2] + (track[3 * f1 + 2] - track[3 * f0 + 2]) * a);
  };

  // ---------------------------------------------------------------- stars (this browser only)
  const loadStars = () => { try { return JSON.parse(localStorage.getItem(STARS_KEY)) || []; } catch (e) { return []; } };
  const saveStars = () => { try { localStorage.setItem(STARS_KEY, JSON.stringify(stars)); } catch (e) { /* private mode: the list lasts for this visit */ } };
  let stars = loadStars().filter(s => sources.some(x => x.id === s.source));
  const isStarred = (id, i) => stars.some(s => s.source === id && s.index === i);

  function renderStars() {
    starsRow.hidden = !stars.length;
    starsRow.replaceChildren(el("span", "replay-stars-label", "Starred"), ...stars.map(s => {
      const src = sources.find(x => x.id === s.source);
      const wrap = el("span", "star-chip");
      const open = el("button", "chip", `★ ${src.title} · #${s.index + 1}`);
      open.type = "button";
      open.title = s.end;
      open.addEventListener("click", () => openEpisode(s.source, s.index));
      const remove = el("button", "star-remove", "×");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove episode ${s.index + 1} from the starred list`);
      remove.addEventListener("click", () => { stars = stars.filter(x => x !== s); saveStars(); renderStars(); renderNav(); });
      wrap.append(open, remove);
      return wrap;
    }));
  }

  function toggleStar() {
    if (!ep) return;
    const i = ep.index;
    if (isStarred(source.id, i)) stars = stars.filter(s => !(s.source === source.id && s.index === i));
    else stars.push({ source: source.id, index: i, end: ep.end });
    saveStars(); renderStars(); renderNav();
  }

  // ---------------------------------------------------------------- browsing
  function renderCategories() {
    cats.replaceChildren(...CATEGORIES.filter(c => lists[c.key].length).map(c => {
      const b = el("button", "chip", `${c.label} · ${lists[c.key].length}`);
      b.type = "button";
      b.dataset.key = c.key;
      b.addEventListener("click", () => show(c.key, 0));
      return b;
    }));
  }

  function renderNav() {
    if (!lists) return;
    const c = CATEGORIES.find(x => x.key === category), list = lists[category];
    where.textContent = `${c.one} ${position + 1} of ${list.length}${c.order ? ` · ${c.order}` : ""}`;
    for (const b of cats.children) b.setAttribute("aria-pressed", String(b.dataset.key === category));
    const i = list[position];
    if (document.activeElement !== numInput) numInput.value = String(i + 1);
    numInput.max = String(ix.episodes_total);
    const starred = isStarred(source.id, i);
    starBtn.textContent = starred ? "★ Starred" : "☆ Star";
    starBtn.setAttribute("aria-pressed", String(starred));
  }

  async function show(cat, pos) {
    const list = lists[cat];
    if (!list || !list.length) return;
    category = cat;
    position = ((pos % list.length) + list.length) % list.length;
    const i = list[position];
    renderNav();
    const mine = ++episodeRequest;
    let data;
    try {
      data = await getEpisode(source, i);
    } catch (err) {
      if (mine === episodeRequest) { msg.textContent = `Could not load episode ${i + 1} (${err.message}).`; msg.hidden = false; }
      return;
    }
    if (mine !== episodeRequest) return;  // a newer choice replaced this one while it loaded
    msg.hidden = true;
    const label = cat === "highlights" ? ix.highlights[position].label : `${CATEGORIES.find(x => x.key === cat).one} ${position + 1} of ${list.length}`;
    setEpisode(data, label);
    try { history.replaceState(null, "", `#replay/${source.id}/${i + 1}`); } catch (e) { /* not essential */ }
    getEpisode(source, list[(position + 1) % list.length]).catch(() => {});  // have the server prepare the next one
  }

  // Open episode i (0-based): in the current category if it contains it, else in All.
  function openIndex(i) {
    const k = lists[category].indexOf(i);
    return k >= 0 ? show(category, k) : show("all", i);
  }

  async function openEpisode(id, i) {
    if (source && source.id === id) return openIndex(i);
    return loadSource(id, i);
  }

  async function loadSource(id, open = null) {
    const s = sources.find(x => x.id === id) || sources[0];
    const mine = ++sourceRequest;
    sourceSel.value = s.id;
    msg.textContent = "Loading…";
    msg.hidden = false;
    let index;
    try {
      index = await getIndex(s);
    } catch (err) {
      if (mine === sourceRequest) msg.textContent = `Could not load the episode list (${err.message}).`;
      return;
    }
    if (mine !== sourceRequest) return;
    source = s; ix = index; lists = buildLists(ix);
    note.textContent = s.note;
    buildArena(ix);
    renderCategories();
    if (open !== null && open >= 0 && open < ix.episodes_total) {
      const h = lists.highlights.indexOf(open);
      category = h >= 0 ? "highlights" : "all";
      await show(category, h >= 0 ? h : open);
    } else {
      await show("highlights", 0);
    }
  }

  // ---------------------------------------------------------------- one episode
  function disposeEpisode() {
    for (const d of drones) {
      d.mesh.material.dispose();
      d.trail.geometry.dispose();
      lineMaterials.delete(d.trail.material); d.trail.material.dispose();
      scene.remove(d.mesh, d.trail);
    }
    for (const p of paths.children) { p.geometry.dispose(); lineMaterials.delete(p.material); p.material.dispose(); }
    paths.clear();
    for (const f of flashes.children) f.material.dispose();
    flashes.clear();
    pillars.clear();
  }

  function setEpisode(data, label) {
    disposeEpisode();
    ep = data;
    for (const [x, y] of ep.pillars) {
      const m = new THREE.Mesh(pillarGeo, pillarMat);
      m.rotation.x = Math.PI / 2;  // CylinderGeometry stands along y; here pillars stand along z
      m.position.set(x, y, ix.arena.center[2]);
      pillars.add(m);
    }
    drones = [{ role: "prey", track: Float32Array.from(ep.prey), down: null }]
      .concat(ep.predators.map((t, k) => ({ role: "predator", track: Float32Array.from(t), down: ep.dead[k], number: k + 1 })));
    for (const d of drones) {
      d.color = ROLE[d.role];
      d.mesh = new THREE.Mesh(droneGeo, new THREE.MeshStandardMaterial({ color: d.color, emissive: d.color, emissiveIntensity: 0.6, roughness: 0.35 }));
      d.trail = new Line2(new LineGeometry(), lineMaterial(3.5));
      d.trail.visible = false;
      scene.add(d.mesh, d.trail);
      // The whole path, faint, for "Full paths": up to the frame where the drone went down.
      const end = d.down === null ? ep.frames : d.down + 1;
      if (end >= 2) {
        const faint = tmpColor.copy(BACKGROUND).lerp(d.color, 0.55);
        const colors = new Float32Array(end * 3);
        for (let j = 0; j < end; j++) colors.set([faint.r, faint.g, faint.b], 3 * j);
        const g = new LineGeometry();
        g.setPositions(d.track.subarray(0, end * 3));
        g.setColors(colors);
        paths.add(new Line2(g, lineMaterial(1.5)));
      }
    }
    // Flashes: where a predator went down, and how the episode ended.
    events = [];
    const flash = (f, pos, color) => {
      const m = new THREE.Mesh(flashGeo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0, depthWrite: false }));
      m.position.copy(pos); m.visible = false;
      flashes.add(m);
      events.push({ frame: f, mesh: m });
    };
    for (const d of drones) if (d.down !== null) flash(d.down, at(d.track, d.down, new THREE.Vector3()), 0xff5a4f);
    const endColor = ep.end.startsWith("caught") ? 0xffffff : ep.end.includes("pillar") ? 0xff5a4f : ep.end.includes("left") ? 0xffb340 : null;
    if (endColor !== null) flash(ep.frames - 1, at(drones[0].track, ep.frames - 1, new THREE.Vector3()), endColor);

    paths.visible = pathBtn.getAttribute("aria-pressed") === "true";
    frame = 0; holdUntil = 0; shownDowns = null;
    slider.max = String(ep.frames - 1);
    // In "All" the label already is the episode number.
    const which = category === "all" ? "" : ` · episode ${ep.index + 1} of ${ix.episodes_total}`;
    info.replaceChildren(el("strong", "", label), el("span", "", `${which} · ${ep.end}`), el("div", "downs"));
    canvas.setAttribute("aria-label", `3D replay: episode ${ep.index + 1}, ${ep.end}`);
    renderNav();
    requestRender();
  }

  // ---------------------------------------------------------------- drawing
  function updateTrail(d) {
    const window = TRAIL_S / ix.dt;
    const head = d.down === null ? frame : Math.min(frame, d.down);
    // The window is in time, so the tail of a downed drone catches up with its head and the trail disappears.
    const start = Math.max(0, Math.ceil(frame - window));
    const whole = Math.floor(head);
    const count = whole - start + 1 + (head > whole ? 1 : 0);
    if (whole < start || count < 2) { d.trail.visible = false; return; }
    const pos = new Float32Array(count * 3), col = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const f = start + i <= whole ? start + i : head;
      at(d.track, f, tmp);
      pos[3 * i] = tmp.x; pos[3 * i + 1] = tmp.y; pos[3 * i + 2] = tmp.z;
      const s = Math.max(0, 1 - (frame - f) / window) ** 1.6;  // 1 at a live drone's head, 0 two seconds back
      tmpColor.copy(BACKGROUND).lerp(d.color, s);
      col[3 * i] = tmpColor.r; col[3 * i + 1] = tmpColor.g; col[3 * i + 2] = tmpColor.b;
    }
    const old = d.trail.geometry;
    d.trail.geometry = new LineGeometry();
    d.trail.geometry.setPositions(pos);
    d.trail.geometry.setColors(col);
    old.dispose();
    d.trail.visible = true;
  }

  function draw() {
    const downs = [];
    for (const d of drones) {
      const isDown = d.down !== null && frame >= d.down;
      at(d.track, isDown ? d.down : frame, d.mesh.position);
      const color = isDown ? ROLE.down : d.color;
      d.mesh.material.color.copy(color);
      d.mesh.material.emissive.copy(color);
      d.mesh.material.emissiveIntensity = isDown ? 0.15 : 0.6;
      updateTrail(d);
      if (isDown) downs.push(`predator ${d.number} down at ${(d.down * ix.dt).toFixed(2)} s`);
    }
    for (const e of events) {
      const age = (frame - e.frame) * ix.dt;
      e.mesh.visible = age >= 0 && age < FLASH_S;
      if (e.mesh.visible) {
        e.mesh.scale.setScalar(0.25 + 1.1 * (age / FLASH_S));
        e.mesh.material.opacity = 0.7 * (1 - age / FLASH_S);
      }
    }
    const text = downs.join(" · ");
    if (text !== shownDowns) { info.querySelector(".downs").textContent = text; shownDowns = text; }
    clock.textContent = `${(frame * ix.dt).toFixed(2)} s / ${((ep.frames - 1) * ix.dt).toFixed(2)} s`;
    if (document.activeElement !== slider) slider.value = String(Math.floor(frame));
    playBtn.textContent = playing ? "Pause" : "Play";
  }

  function advance(wallS, now) {
    const lastFrame = ep.frames - 1;
    if (frame >= lastFrame) {
      if (!holdUntil) holdUntil = now + HOLD_S * 1000;
      else if (now >= holdUntil) { frame = 0; holdUntil = 0; }
      return;
    }
    frame = Math.min(lastFrame, frame + (wallS * speed) / ix.dt);
  }

  // The stage's own position decides when the observer's state is not settled yet (e.g. right after mounting).
  const onScreen = () => {
    const r = stage.getBoundingClientRect();
    return document.visibilityState === "visible" && r.width > 0 && r.bottom > 0 && r.top < innerHeight;
  };

  function requestRender() {
    if (raf || ticking) return;
    if (!visible) visible = onScreen();
    if (!visible) return;
    last = performance.now();
    raf = requestAnimationFrame(tick);
  }

  function tick(now) {
    raf = 0;
    ticking = true;
    let again = false;
    try {
      const wallS = Math.min((now - last) / 1000, 0.1);
      last = now;
      let following = false;
      if (ep) {
        if (playing) advance(wallS, now);
        if (follow) {  // keep the camera's offset and move the target along with the prey
          const delta = at(drones[0].track, frame, tmp).sub(controls.target).multiplyScalar(0.12);
          controls.target.add(delta);
          camera.position.add(delta);
          following = delta.lengthSq() > 1e-6;
        }
        draw();
      }
      const moving = controls.update();
      renderer.render(scene, camera);
      again = visible && (playing || moving || following);
    } finally {
      ticking = false;  // an error in one frame must not block every later frame
    }
    if (again) raf = requestAnimationFrame(tick);
  }

  function setView(key) {
    follow = key === "follow";
    if (!follow) {
      camera.position.set(...VIEWS[key].eye);
      controls.target.set(...VIEWS[key].target);
    }
    for (const b of views.children) b.setAttribute("aria-pressed", String(b.dataset.view === key));
    requestRender();
  }

  // Keep the horizontal field of view of the video camera whatever the shape of the box.
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(H_FOV / 2)) / camera.aspect));
    camera.updateProjectionMatrix();
    for (const m of lineMaterials) m.resolution.set(w, h);
    requestRender();
  }

  // ---------------------------------------------------------------- events
  prevBtn.addEventListener("click", () => show(category, position - 1));
  nextBtn.addEventListener("click", () => show(category, position + 1));
  randomBtn.addEventListener("click", () => {
    const n = lists ? lists[category].length : 0;
    if (n > 1) show(category, (position + 1 + Math.floor(Math.random() * (n - 1))) % n);  // never the one shown
  });
  starBtn.addEventListener("click", toggleStar);
  numInput.addEventListener("change", () => {
    const i = Number(numInput.value) - 1;
    if (ix && Number.isInteger(i) && i >= 0 && i < ix.episodes_total) openIndex(i);
  });
  playBtn.addEventListener("click", () => {
    if (!playing && ep && frame >= ep.frames - 1) frame = 0;
    playing = !playing; holdUntil = 0;
    playBtn.textContent = playing ? "Pause" : "Play";
    requestRender();
  });
  slider.addEventListener("input", () => { frame = Number(slider.value); holdUntil = 0; requestRender(); });
  speedSel.addEventListener("change", () => { speed = Number(speedSel.value); });
  pathBtn.addEventListener("click", () => {
    const on = pathBtn.getAttribute("aria-pressed") !== "true";
    pathBtn.setAttribute("aria-pressed", String(on));
    paths.visible = on;
    requestRender();
  });
  views.addEventListener("click", ev => { const b = ev.target.closest("[data-view]"); if (b) setView(b.dataset.view); });
  sourceSel.addEventListener("change", () => loadSource(sourceSel.value));
  canvas.addEventListener("keydown", ev => {
    if (!ep) return;
    if (ev.key === " ") { playBtn.click(); ev.preventDefault(); }
    if (ev.key === "n" || ev.key === "p") show(category, position + (ev.key === "n" ? 1 : -1));
    if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
      playing = false;
      playBtn.textContent = "Play";
      frame = Math.max(0, Math.min(ep.frames - 1, Math.round(frame) + (ev.key === "ArrowRight" ? 1 : -1)));
      requestRender();
      ev.preventDefault();
    }
  });
  controls.addEventListener("change", requestRender);
  new ResizeObserver(resize).observe(stage);
  // Draw only while the stage is on screen; requestAnimationFrame already pauses in background tabs. One callback
  // can carry several entries (e.g. "off screen" and then "on screen" during a smooth scroll): the last one is current.
  new IntersectionObserver(entries => {
    visible = entries[entries.length - 1].isIntersecting;
    requestRender();
  }).observe(stage);

  // Links: #replay/<source>/<episode number> opens that episode.
  const fromHash = () => {
    const m = /^#replay\/([\w-]+)\/(\d+)$/.exec(location.hash);
    return m && sources.some(s => s.id === m[1]) ? { id: m[1], index: Number(m[2]) - 1 } : null;
  };
  addEventListener("hashchange", () => { const h = fromHash(); if (h) openEpisode(h.id, h.index); });

  setView("video");
  resize();
  renderStars();
  const start = fromHash();
  loadSource(start ? start.id : sources[0].id, start ? start.index : null);
}
