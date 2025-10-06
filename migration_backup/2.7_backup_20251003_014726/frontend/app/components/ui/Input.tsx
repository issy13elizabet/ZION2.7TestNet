import React from 'react'
import { clsx } from 'clsx'

type Props = React.InputHTMLAttributes<HTMLInputElement> & {
  invalid?: boolean
}

export default function Input({ className, invalid, ...rest }: Props) {
  return (
    <input
      className={clsx(
        'w-full rounded-md border bg-white px-3 py-2 text-sm text-slate-900 shadow-sm placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-500/40',
        'dark:bg-slate-900 dark:text-slate-100 dark:placeholder:text-slate-500',
        invalid ? 'border-red-500' : 'border-slate-300 dark:border-slate-700',
        className,
      )}
      {...rest}
    />
  )
}
