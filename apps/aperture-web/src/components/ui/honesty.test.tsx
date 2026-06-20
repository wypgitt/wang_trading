import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ComingState, DataUnavailable, DataState } from './honesty';

describe('ComingState', () => {
  it('renders the title, the verbatim unlock condition, and the wave', () => {
    render(<ComingState title="Portfolio value — coming" unlock="Unlocks when the engine persists positions." wave={5} />);
    expect(screen.getByText('Portfolio value — coming')).toBeInTheDocument();
    expect(screen.getByText(/Unlocks when the engine persists positions/)).toBeInTheDocument();
    expect(screen.getByText('Wave 5')).toBeInTheDocument();
  });
});

describe('DataUnavailable', () => {
  it('reads "Coming" for a roadmap-gated field', () => {
    render(<DataUnavailable unlock="coming later" />);
    expect(screen.getByText('Coming')).toBeInTheDocument();
  });
  it('reads "Model?" for a model-gated field', () => {
    render(<DataUnavailable unlock="load a model" modelGated />);
    expect(screen.getByText('Model?')).toBeInTheDocument();
  });
});

describe('DataState — the seven-state switch', () => {
  it('shows the live value when live', () => {
    render(<DataState state="live">0.71</DataState>);
    expect(screen.getByText('0.71')).toBeInTheDocument();
  });
  it('uses the em-dash ONLY for a legitimately-empty live value', () => {
    render(<DataState state="empty" />);
    expect(screen.getByText('—')).toBeInTheDocument();
  });
  it('never renders a value for an ABSENT field — routes to Coming', () => {
    render(<DataState state="coming" unlock="not yet" />);
    expect(screen.getByText('Coming')).toBeInTheDocument();
    expect(screen.queryByText('—')).not.toBeInTheDocument();
  });
  it('routes model-gated to the Model? affordance', () => {
    render(<DataState state="modelGated" />);
    expect(screen.getByText('Model?')).toBeInTheDocument();
  });
  it('renders an inline error with the copyable request id', () => {
    render(<DataState state="error" label="ideas" requestId="req_abc" />);
    expect(screen.getByText(/Couldn’t load/)).toBeInTheDocument();
    expect(screen.getByText(/req_abc/)).toBeInTheDocument();
  });
});
