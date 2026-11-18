/* Order-entry work task: the pure half (no DOM), so it can be checked without
 * a browser. The DOM half is dataentry-task.js.
 *
 * WHY THIS TASK EXISTS. The SART and the arithmetic drill are lab tasks: short,
 * obviously "a test", and done in a tab the participant has just been told is
 * watching their attention. Under those conditions almost nobody drifts - the
 * demand characteristics swamp the effect being measured. Order entry is
 * ordinary office work: self-paced, never finished, mildly tedious, and the
 * kind of thing people really do alongside their phone and other tabs. That
 * makes distraction something that can happen rather than something the
 * participant is primed to suppress.
 *
 * Scoring stays in the arithmetic shape (attempted / correct) on purpose, so
 * kind=study&task_type=dataentry exports straight into analyze_study.py's
 * existing accuracy / correct_per_min measures with no new columns.
 */

import { seeded, randInt } from "./rng.js";

export const TASK_TYPE = "dataentry";
export const TASK_VERSION = "dataentry-1.0";

/* Fields in entry order. `normalize` decides what counts as the same answer:
 * the task measures attention, not typing pedantry, so case and stray spaces
 * never make an entry wrong, but a wrong digit always does. */
export const FIELDS = [
  { key: "customer", label: "Customer", normalize: (v) => v.trim().replace(/\s+/g, " ").toLowerCase() },
  { key: "date", label: "Order date", normalize: (v) => v.trim().replace(/\s+/g, "") },
  { key: "item", label: "Item code", normalize: (v) => v.trim().replace(/\s+/g, "").toUpperCase() },
  { key: "qty", label: "Quantity", normalize: (v) => String(Number.parseInt(v.trim(), 10)) },
  {
    key: "price", label: "Unit price", normalize: (v) => {
      const n = Number.parseFloat(v.trim().replace(/^\$/, "").replace(/,/g, ""));
      return Number.isFinite(n) ? n.toFixed(2) : "";
    },
  },
];

const FIRST = ["Maria", "James", "Aisha", "Daniel", "Priya", "Kevin", "Sofia", "Marcus",
  "Elena", "Tyler", "Grace", "Omar", "Hannah", "Luis", "Chloe", "Ethan", "Fatima",
  "Noah", "Mei", "Caleb", "Rosa", "Jordan", "Nadia", "Victor"];
const LAST = ["Alvarez", "Brennan", "Chowdhury", "Delgado", "Eriksen", "Fontaine",
  "Gallagher", "Hashemi", "Ibarra", "Jankowski", "Kowalczyk", "Lindqvist", "Morales",
  "Nakamura", "Okafor", "Petrovic", "Quintero", "Rasmussen", "Sandoval", "Thibodeaux",
  "Underwood", "Vasquez", "Whitfield", "Yamamoto"];
const PREFIX = ["KT", "BR", "MX", "PL", "HD", "SV", "TR", "GN"];
const DESCR = ["Hex bolts, zinc", "Cable ties, 8 in", "Shelf bracket", "Hinge, brushed",
  "Wood screws, #8", "Drawer slide pair", "Door stop, rubber", "Wall anchor kit",
  "Washer assortment", "Corner brace", "Caster wheel", "Pipe clamp"];

const pad = (n) => String(n).padStart(2, "0");

/* Deterministic per (participant, condition), like every other task's stimuli,
 * and endless: a self-paced task must never run out before the clock does. */
export class SlipSource {
  constructor(participant, condition) {
    this.rand = seeded(`${participant}-${condition}-dataentry`);
  }

  _pick(list) { return list[randInt(this.rand, 0, list.length - 1)]; }

  next() {
    const r = this.rand;
    const month = randInt(r, 1, 12);
    const day = randInt(r, 1, 28);
    const cents = randInt(r, 149, 18999);
    return {
      slip: 40000 + randInt(r, 100, 9999),
      customer: `${this._pick(FIRST)} ${this._pick(LAST)}`,
      date: `${pad(month)}/${pad(day)}/2026`,
      item: `${this._pick(PREFIX)}-${randInt(r, 1000, 9999)}`,
      description: this._pick(DESCR),
      qty: String(randInt(r, 1, 48)),
      price: (cents / 100).toFixed(2),
    };
  }
}

/* The field keys the participant got wrong (empty = a correct entry). */
export function checkEntry(slip, entry) {
  return FIELDS
    .filter((f) => f.normalize(String(entry[f.key] ?? "")) !== f.normalize(String(slip[f.key])))
    .map((f) => f.key);
}
