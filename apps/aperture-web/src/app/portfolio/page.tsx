import { ComingScreen } from '@/components/ComingScreen';
import { screenById } from '@/lib/readiness';

export default function Page() {
  return <ComingScreen spec={screenById('portfolio')!} />;
}
