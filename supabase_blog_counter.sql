create table if not exists public.blog_generation_counter (
    id boolean primary key default true check (id is true),
    total bigint not null default 0 check (total >= 0)
);

insert into public.blog_generation_counter (id, total)
values (true, 0)
on conflict (id) do nothing;

alter table public.blog_generation_counter enable row level security;

revoke all on table public.blog_generation_counter from anon, authenticated;
grant select, insert, update on table public.blog_generation_counter to service_role;

create or replace function public.get_blog_generation_count()
returns bigint
language sql
stable
security invoker
set search_path = ''
as $$
    select total
    from public.blog_generation_counter
    where id is true;
$$;

create or replace function public.increment_blog_generation_count()
returns bigint
language sql
security invoker
set search_path = ''
as $$
    insert into public.blog_generation_counter as counter (id, total)
    values (true, 1)
    on conflict (id) do update
    set total = counter.total + 1
    returning total;
$$;

revoke all on function public.get_blog_generation_count() from public, anon, authenticated;
revoke all on function public.increment_blog_generation_count() from public, anon, authenticated;
grant execute on function public.get_blog_generation_count() to service_role;
grant execute on function public.increment_blog_generation_count() to service_role;
