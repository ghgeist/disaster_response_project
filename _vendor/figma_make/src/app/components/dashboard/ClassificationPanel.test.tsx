import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { ClassificationPanel } from './ClassificationPanel'

describe('ClassificationPanel', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            categories: [
              { name: 'Water', confidence: 0.92, volume: 100 },
            ],
            severity: 'HIGH',
          }),
      })
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders with Classify Message header and Run Classification button', () => {
    render(<ClassificationPanel />)
    expect(screen.getByText(/Classify Message/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Run Classification/ })).toBeInTheDocument()
  })

  it('shows error message when classify fetch fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new Error('Classification 503'))
    )
    render(<ClassificationPanel />)
    fireEvent.change(screen.getByPlaceholderText(/Paste a raw message/), {
      target: { value: 'Need water and shelter' },
    })
    fireEvent.click(screen.getByRole('button', { name: /Run Classification/ }))
    await waitFor(() => {
      expect(screen.getByText(/Classification 503/)).toBeInTheDocument()
    })
  })

  it('shows no-match state when API returns empty categories', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({ categories: [], severity: 'LOW' }),
      })
    )
    render(<ClassificationPanel />)
    fireEvent.change(screen.getByPlaceholderText(/Paste a raw message/), {
      target: { value: 'Hello world' },
    })
    fireEvent.click(screen.getByRole('button', { name: /Run Classification/ }))
    await waitFor(() => {
      expect(screen.getByText(/No emergency categories detected/)).toBeInTheDocument()
    })
    expect(screen.getByText('NO MATCH')).toBeInTheDocument()
  })
})
