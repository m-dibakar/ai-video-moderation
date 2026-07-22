import { render, screen } from '@testing-library/react';
import App from './App';

test('renders hero headline and analyzer entry point', () => {
  render(<App />);
  expect(screen.getByText(/missing them/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /run analysis/i })).toBeDisabled();
});
