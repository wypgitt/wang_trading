import { IPhoneFrame } from './IPhoneFrame';
import { IOSApp } from './IOSApp';

export function IOSStage() {
  return (
    <div
      style={{
        height: '100%',
        overflow: 'auto',
        display: 'grid',
        placeItems: 'center',
        background: 'radial-gradient(1100px 560px at 50% -8%, #161a22, #0a0c10 70%)',
      }}
    >
      <div style={{ padding: '20px 0' }}>
        <IPhoneFrame>
          <IOSApp />
        </IPhoneFrame>
      </div>
    </div>
  );
}
