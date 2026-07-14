import { copyFile, mkdir, rm } from 'node:fs/promises';
import { resolve } from 'node:path';

const projectRoot = resolve(import.meta.dirname, '..');
const outputDirectory = resolve(projectRoot, 'dist');

await rm(outputDirectory, { recursive: true, force: true });
await mkdir(outputDirectory, { recursive: true });
await copyFile(
  resolve(projectRoot, 'templates', 'index.html'),
  resolve(outputDirectory, 'index.html'),
);

console.log('Archive AI frontend built in dist/.');
