import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ActionPill, Delta } from './primitives';

describe('ActionPill', () => {
  it('maps engine actions to human labels', () => {
    const { rerender } = render(<ActionPill action="BUY" />);
    expect(screen.getByText('Buy')).toBeInTheDocument();
    rerender(<ActionPill action="SELL" />);
    expect(screen.getByText('Sell')).toBeInTheDocument();
    rerender(<ActionPill action="MODEL_REQUIRED" />);
    expect(screen.getByText('Model?')).toBeInTheDocument();
    rerender(<ActionPill action="NO_DATA" />);
    expect(screen.getByText('No data')).toBeInTheDocument();
  });
});

describe('Delta', () => {
  it('colours direction and is never colour-alone (arrow + sign)', () => {
    const { container } = render(<Delta value={0.092} dp={1} />);
    const el = container.querySelector('span')!;
    expect(el.className).toContain('pos');
    expect(el.textContent).toContain('▲');
    expect(el.textContent).toContain('+9.2%');
  });
  it('marks negatives', () => {
    const { container } = render(<Delta value={-0.052} dp={1} />);
    const el = container.querySelector('span')!;
    expect(el.className).toContain('neg');
    expect(el.textContent).toContain('▼');
  });
});
