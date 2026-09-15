import { useNavigate } from 'react-router-dom';
import { Link } from 'react-router-dom';

import { SidebarHistory } from '@/components/sidebar-history';
import { SidebarUserNav } from '@/components/sidebar-user-nav';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from '@/components/ui/sidebar';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { DbIcon } from '@/components/ui/db-icon';
import {
  NewChatIcon,
  SidebarCollapseIcon,
  SidebarExpandIcon,
} from '@/components/icons';
import { cn } from '@/lib/utils';
import type { ClientSession } from '@chat-template/auth';
import { Action } from './elements/actions';
import { useToolVisibility } from '@/contexts/ToolVisibilityContext';

export function AppSidebar({
  user,
  preferredUsername,
}: {
  user: ClientSession['user'] | undefined;
  preferredUsername: string | null;
}) {
  const navigate = useNavigate();
  const { setOpenMobile, open, openMobile, isMobile, toggleSidebar } =
    useSidebar();
  const { showToolCalls, setShowToolCalls } = useToolVisibility();

  const effectiveOpen = open || (isMobile && openMobile);

  return (
    <Sidebar collapsible="icon" className="group-data-[side=left]:border-r-0">
      {/* ── Header: app title + collapse toggle ────────────────────────── */}
      <SidebarHeader
        className={cn(
          'h-[44px] flex-row items-center gap-2 px-2 py-0',
          effectiveOpen ? 'justify-between' : 'justify-center',
        )}
      >
        {effectiveOpen && (
          <Link
            to="/"
            onClick={() => setOpenMobile(false)}
            className="flex items-center overflow-hidden px-1"
          >
            <span className="font-semibold text-base text-foreground">
              Chatbot
            </span>
          </Link>
        )}

        <Action
          onClick={toggleSidebar}
          tooltip={effectiveOpen ? 'Collapse sidebar' : 'Expand sidebar'}
        >
          <DbIcon
            icon={effectiveOpen ? SidebarCollapseIcon : SidebarExpandIcon}
            size={16}
            color="muted"
          />
        </Action>
      </SidebarHeader>

      {/* ── Nav: New Chat item ───────────────────────────────────────────── */}
      <div className="px-2 pt-2">
        <SidebarMenu>
          <SidebarMenuItem>
            <Tooltip>
              <TooltipTrigger asChild>
                <SidebarMenuButton
                  type="button"
                  className="h-8 cursor-pointer p-1 md:p-2"
                  onClick={() => {
                    setOpenMobile(false);
                    navigate('/');
                  }}
                >
                  <DbIcon icon={NewChatIcon} size={16} color="default" />
                  <span className="group-data-[collapsible=icon]:hidden">
                    New chat
                  </span>
                </SidebarMenuButton>
              </TooltipTrigger>
              <TooltipContent
                side="right"
                style={{ display: open ? 'none' : 'block' }}
              >
                New chat
              </TooltipContent>
            </Tooltip>
          </SidebarMenuItem>
        </SidebarMenu>
      </div>

      {/* ── Chat history ────────────────────────────────────────────────── */}
      <SidebarContent>
        {effectiveOpen && <SidebarHistory user={user} />}
      </SidebarContent>

      {/* ── User nav ────────────────────────────────────────────────────── */}
      <SidebarFooter>
        {effectiveOpen && (
          <label className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 text-muted-foreground text-sm hover:bg-sidebar-accent hover:text-sidebar-accent-foreground">
            <input
              type="checkbox"
              checked={showToolCalls}
              onChange={(event) => setShowToolCalls(event.target.checked)}
              className="size-4 cursor-pointer accent-primary"
              data-testid="show-tool-calls"
            />
            <span>Show tool calls</span>
          </label>
        )}
        {user && (
          <SidebarUserNav user={user} preferredUsername={preferredUsername} />
        )}
      </SidebarFooter>
    </Sidebar>
  );
}
