#!/usr/bin/env node
import { createHash, randomBytes, randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'

const args = process.argv.slice(2)
const command = args.shift()
const file = resolve(option('--file') ?? process.env.MCP_ACCESS_KEYS_FILE ?? '/data/access-keys.json')

function option(name) {
  const index = args.indexOf(name)
  if (index === -1) return undefined
  const value = args[index + 1]
  if (!value || value.startsWith('--')) throw new Error(`${name} requires a value.`)
  args.splice(index, 2)
  return value
}

function integerOption(name, fallback) {
  const raw = option(name)
  const value = raw === undefined ? fallback : Number(raw)
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer.`)
  return value
}

function tokenHash(token) {
  return `sha256:${createHash('sha256').update(token, 'utf8').digest('hex')}`
}

function normalizeId(input) {
  const id = input.toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 48)
  if (!id) throw new Error('Cannot derive a valid key id.')
  return id
}

async function load() {
  if (!existsSync(file)) return { version: 1, keys: [] }
  const parsed = JSON.parse(await readFile(file, 'utf8'))
  if (parsed.version !== 1 || !Array.isArray(parsed.keys)) throw new Error('Invalid access key file.')
  return parsed
}

async function save(data) {
  await mkdir(dirname(file), { recursive: true, mode: 0o700 })
  const temporary = `${file}.${process.pid}.tmp`
  await writeFile(temporary, `${JSON.stringify(data, null, 2)}\n`, { mode: 0o600 })
  await rename(temporary, file)
}

function find(data, id) {
  const record = data.keys.find((candidate) => candidate.id === id)
  if (!record) throw new Error(`Unknown access key id: ${id}`)
  return record
}

async function create() {
  const name = option('--name')
  if (!name || name.length > 128) throw new Error('--name is required and must be at most 128 characters.')
  const dailyLimit = integerOption('--daily-limit', 200)
  const requestedId = option('--id')
  const data = await load()
  const baseId = normalizeId(requestedId ?? name)
  let id = requestedId ? baseId : `${baseId}-${randomUUID().slice(0, 8)}`
  if (data.keys.some((candidate) => candidate.id === id)) throw new Error(`Access key id already exists: ${id}`)
  const token = `atv_${id}_${randomBytes(32).toString('base64url')}`
  data.keys.push({
    id,
    name,
    tokenHash: tokenHash(token),
    enabled: true,
    dailyLimit,
    createdAt: new Date().toISOString(),
  })
  await save(data)
  console.log(`Created access key ${id} (${name}), daily limit ${dailyLimit}.`)
  console.log('Copy this token now; it will not be shown again:')
  console.log(token)
}

async function importToken() {
  const id = normalizeId(option('--id') ?? '')
  const name = option('--name') ?? id
  const tokenEnvironment = option('--token-env')
  if (!tokenEnvironment) throw new Error('--token-env is required; plaintext tokens are never accepted as command arguments.')
  const token = process.env[tokenEnvironment]
  if (!token || token.length < 24) throw new Error(`Environment variable ${tokenEnvironment} is missing or too short.`)
  const dailyLimit = integerOption('--daily-limit', 2_000)
  const data = await load()
  if (data.keys.some((candidate) => candidate.id === id)) throw new Error(`Access key id already exists: ${id}`)
  data.keys.push({
    id,
    name,
    tokenHash: tokenHash(token),
    enabled: true,
    dailyLimit,
    createdAt: new Date().toISOString(),
  })
  await save(data)
  console.log(`Imported access key ${id} (${name}), daily limit ${dailyLimit}. Token was not printed.`)
}

async function update(enabled) {
  const id = args.shift()
  if (!id) throw new Error('An access key id is required.')
  const data = await load()
  find(data, id).enabled = enabled
  await save(data)
  console.log(`${enabled ? 'Enabled' : 'Revoked'} access key ${id}.`)
}

async function setLimit() {
  const id = args.shift()
  if (!id) throw new Error('An access key id is required.')
  const dailyLimit = integerOption('--daily-limit', 0)
  const data = await load()
  find(data, id).dailyLimit = dailyLimit
  await save(data)
  console.log(`Set ${id} daily limit to ${dailyLimit}.`)
}

async function list() {
  const data = await load()
  console.table(data.keys.map(({ id, name, enabled, dailyLimit, createdAt, expiresAt }) => ({
    id, name, enabled, dailyLimit, createdAt, expiresAt: expiresAt ?? '',
  })))
}

async function main() {
  if (command === 'create') return create()
  if (command === 'import') return importToken()
  if (command === 'revoke') return update(false)
  if (command === 'enable') return update(true)
  if (command === 'set-limit') return setLimit()
  if (command === 'list') return list()
  throw new Error('Usage: manage-access.mjs <create|import|revoke|enable|set-limit|list> [options]')
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error)
  process.exit(1)
})

