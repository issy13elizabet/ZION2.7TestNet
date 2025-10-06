import React from 'react'
import { clsx } from 'clsx'

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'default' | 'danger' | 'ghost'
  iconLeft?: React.ReactNode
  iconRight?: React.ReactNode
}

export default function Button({ className, variant = 'default', iconLeft, iconRight, children, ...rest }: Props) {
  const base = 'inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition active:scale-[.99] focus:outline-none focus:ring-2 focus:ring-brand-500/30'
  const variants: Record<string, string> = {
    default: 'border border-slate-300 bg-white text-slate-900 shadow-sm hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100',
    primary: 'border border-brand-600 bg-brand-600 text-white shadow-sm hover:bg-brand-700',
    danger: 'border border-red-600 bg-red-600 text-white shadow-sm hover:bg-red-700',
    ghost: 'border border-transparent bg-transparent text-slate-900 hover:bg-slate-100 dark:text-slate-100 dark:hover:bg-slate-800',
  }
  return (
    <button className={clsx(base, variants[variant], className)} {...rest}>
      {iconLeft && <span className="-ml-1">{iconLeft}</span>}
      {children}
      {iconRight && <span className="-mr-1">{iconRight}</span>}
    </button>
  )
}
