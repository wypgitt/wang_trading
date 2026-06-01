import { CSSProperties, ReactNode } from 'react';

export function Panel({
  title,
  subtitle,
  action,
  children,
  pad = 18,
  style,
  bodyStyle,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  pad?: number;
  style?: CSSProperties;
  bodyStyle?: CSSProperties;
}) {
  return (
    <div className="card" style={{ padding: pad, ...style }}>
      {(title || action) && (
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, marginBottom: subtitle ? 4 : 14 }}>
          <div>
            {title && <h3 style={{ fontSize: 15 }}>{title}</h3>}
            {subtitle && <div className="muted" style={{ fontSize: 12.5, marginTop: 3 }}>{subtitle}</div>}
          </div>
          {action}
        </div>
      )}
      {subtitle && !title ? null : subtitle && <div style={{ height: 12 }} />}
      <div style={bodyStyle}>{children}</div>
    </div>
  );
}
