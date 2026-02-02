import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MetricsPanel } from './MetricsPanel'

describe('MetricsPanel', () => {
  const _originalFetch = globalThis.fetch

  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            volToday: 1000,
            flaggedRate: 4.2,
            trendData: [
              { time: '6h ago', count: 45 },
              { time: 'Now', count: 60 },
            ],
            topCategories: [
              { name: 'Medical Help', count: 120 },
              { name: 'Water', count: 80 },
            ],
          }),
      })
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders and shows loading state while fetch is pending', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    render(<MetricsPanel />)
    expect(screen.getByText(/Loading metrics/)).toBeInTheDocument()
  })

  it('shows Vol Today and LIVE badge after fetch succeeds', async () => {
    render(<MetricsPanel />)
    await waitFor(() => {
      expect(screen.getByText(/Vol Today/)).toBeInTheDocument()
    })
    expect(screen.getByText('LIVE')).toBeInTheDocument()
    expect(screen.getByText('1,000')).toBeInTheDocument()
  })

  it('shows error message when fetch fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new Error('Metrics 500'))
    )
    render(<MetricsPanel />)
    await waitFor(() => {
      expect(screen.getByText(/Metrics 500/)).toBeInTheDocument()
    })
  })

  it('renders without Infinity/NaN when top category counts are zero', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            volToday: 0,
            flaggedRate: 0,
            trendData: [
              { time: '6h ago', count: 0 },
              { time: 'Now', count: 0 },
            ],
            topCategories: [
              { name: 'Medical Help', count: 0 },
              { name: 'Water', count: 0 },
            ],
          }),
      })
    )
    const { container } = render(<MetricsPanel />)
    await waitFor(() => {
      expect(screen.getByText(/Vol Today/)).toBeInTheDocument()
    })
    expect(screen.getByText('Medical Help')).toBeInTheDocument()
    expect(screen.getByText('Water')).toBeInTheDocument()
    const html = container.innerHTML
    expect(html).not.toMatch(/Infinity/)
    expect(html).not.toMatch(/NaN/)
  })
})
