/**
 * ESLint flat config for Storm Signal Dashboard (React + TypeScript).
 * Focus: ship-friendly with idiot insurance — catch real bugs, avoid style wars.
 *
 * Rule set:
 * - ESLint + TypeScript ESLint recommended (base)
 * - React + React Hooks: rules-of-hooks (error), exhaustive-deps (warn)
 * - Unused vars/args (error; prefix with _ to ignore)
 * - no-explicit-any, no-non-null-assertion (warn)
 * - react/prop-types off (TypeScript handles props)
 * - react/no-unescaped-entities (warn)
 * - eslint-config-prettier last (no style conflicts when Prettier is added)
 */
import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import globals from 'globals';
import eslintConfigPrettier from 'eslint-config-prettier';

const tseslintRecommended = tseslint.configs.recommended;
const tseslintConfigs = Array.isArray(tseslintRecommended)
  ? tseslintRecommended
  : [tseslintRecommended];

export default [
  eslint.configs.recommended,
  ...tseslintConfigs,
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      globals: { ...globals.browser },
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    plugins: {
      react,
      'react-hooks': reactHooks,
    },
    settings: {
      react: { version: '18' },
    },
    rules: {
      ...react.configs.recommended.rules,
      ...react.configs['jsx-runtime'].rules,
      ...reactHooks.configs.recommended.rules,
      // Idiot insurance: catch real bugs without blocking shipping
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-non-null-assertion': 'warn',
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      'no-debugger': 'error',
      // TypeScript handles props; prop-types redundant
      'react/prop-types': 'off',
      // Unescaped entities: warn only (style/readability, not correctness)
      'react/no-unescaped-entities': 'warn',
      // Relaxations for shipping: allow console
      'no-console': 'off',
    },
  },
  eslintConfigPrettier,
];
