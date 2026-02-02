import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { FeedPanel } from './FeedPanel'
import type { SignalItem } from '@/app/data'

const noop = () => {}

function makeSignal(overrides: Partial<SignalItem> = {}): SignalItem {
  return {
    id: 'SIG-1',
    timestamp: new Date('2026-02-01T14:00:00Z'),
    source: 'Direct Report',
    content: 'Test message',
    language: 'en',
    riskLevel: 'LOW',
    categories: [],
    classifications: [],
    isTranslated: false,
    ...overrides,
  }
}

describe('FeedPanel', () => {
  it('renders with empty signals and shows Live Feed header and no-signals message', () => {
    render(
      <FeedPanel
        signals={[]}
        selectedFilters={[]}
        onToggleFilter={noop}
        onClearFilters={noop}
      />
    )
    expect(screen.getByText(/Live Feed/)).toBeInTheDocument()
    expect(screen.getByText(/No signals found/)).toBeInTheDocument()
  })

  it('displays 0% when signal has empty classifications (no -Infinity%)', () => {
    const signals: SignalItem[] = [
      makeSignal({
        id: 'SIG-empty',
        content: 'Information about the National Palace-',
        categories: ['Medical Help', 'Shelter', 'Fire'],
        classifications: [], // all below threshold → empty; Math.max(...[]) would be -Infinity
      }),
    ]
    render(
      <FeedPanel
        signals={signals}
        selectedFilters={[]}
        onToggleFilter={noop}
        onClearFilters={noop}
      />
    )
    expect(screen.getByText('0%')).toBeInTheDocument()
    expect(screen.queryByText(/-Infinity%/)).not.toBeInTheDocument()
  })

  it('displays max confidence when signal has classifications', () => {
    const signals: SignalItem[] = [
      makeSignal({
        classifications: [
          { category: 'Water', confidence: 0.92 },
          { category: 'Shelter', confidence: 0.6 },
        ],
      }),
    ]
    render(
      <FeedPanel
        signals={signals}
        selectedFilters={[]}
        onToggleFilter={noop}
        onClearFilters={noop}
      />
    )
    expect(screen.getByText('92%')).toBeInTheDocument()
  })

  it('shows loading state when loading is true', () => {
    render(
      <FeedPanel
        signals={[]}
        selectedFilters={[]}
        onToggleFilter={noop}
        onClearFilters={noop}
        loading={true}
      />
    )
    expect(screen.getByText(/Loading feed/)).toBeInTheDocument()
  })

  it('shows error message when error is set', () => {
    render(
      <FeedPanel
        signals={[]}
        selectedFilters={[]}
        onToggleFilter={noop}
        onClearFilters={noop}
        error="Feed 500"
      />
    )
    expect(screen.getByText('Feed 500')).toBeInTheDocument()
  })

  it('renders signals in the provided order', () => {
    const signals: SignalItem[] = [
      makeSignal({
        id: 'MANUAL-1',
        content: 'Manual dispatch message',
        timestamp: new Date('2026-02-01T14:05:00Z'),
      }),
      makeSignal({
        id: 'SIG-2',
        content: 'Direct report message',
        timestamp: new Date('2026-02-01T14:00:00Z'),
      }),
    ]
    render(
      <FeedPanel
        signals={signals}
        selectedFilters={[]}
        onToggleFilter={noop}
        onClearFilters={noop}
      />
    )
    const manual = screen.getByText('Manual dispatch message')
    const direct = screen.getByText('Direct report message')
    const position = manual.compareDocumentPosition(direct)
    expect(position & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })
})
