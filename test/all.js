/**
 * Test manifest. Import order is irrelevant (importing only registers suites),
 * but keep it grouped so the report reads top-down through the physics stack.
 */
import { runAndReport } from "./harness.js";

import "./harness.selftest.js";
import "./polar.test.js";
import "./sensor.test.js";
import "./frame.test.js";
import "./tof.test.js";
import "./spoofer.test.js";
import "./intensity.test.js";
import "./inject.test.js";
import "./remove.test.js";
import "./realizability.test.js";
import "./io.test.js";
import "./phantom.test.js";
import "./iou.test.js";
import "./detect.test.js";
import "./export.test.js";
import "./atlas-removal.test.js";
import "./sweep.test.js";
import "./atlas-inject.test.js";
import "./pose.test.js";
import "./schedule.test.js";
import "./attacker.test.js";
import "./atlas-constants.test.js";

await runAndReport();
