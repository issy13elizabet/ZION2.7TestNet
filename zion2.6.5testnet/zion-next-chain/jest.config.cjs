/** @type {import('jest').Config} */
export default {
  testEnvironment: 'node',
  transform: {
    '^.+\\.(ts|tsx)$': ['ts-jest', { useESM: true }],
  },
  moduleFileExtensions: ['ts','js','cjs','mjs','json'],
  extensionsToTreatAsEsm: ['.ts'],
  roots: ['<rootDir>/tests'],
  verbose: true,
};
