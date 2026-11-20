import { test, expect } from '../fixtures';
import { ChatPage } from '../pages/chat';

const chatId = '11111111-1111-4111-8111-111111111111';
const messageId = '22222222-2222-4222-8222-222222222222';

test('new ephemeral chat hides feedback even when a trace is available', async ({
  adaContext,
}) => {
  const { page } = adaContext;
  const feedbackRequests: string[] = [];
  await page.route('**/api/config', (route) =>
    route.fulfill({ json: { features: { chatHistory: false, feedback: true } } }),
  );
  await page.route('**/api/feedback**', async (route) => {
    feedbackRequests.push(route.request().url());
    await route.fulfill({ json: {} });
  });
  const events = [
    { type: 'start', messageId },
    { type: 'text-start', id: 'text-1' },
    { type: 'text-delta', id: 'text-1', delta: 'Streamed response' },
    { type: 'text-end', id: 'text-1' },
    { type: 'data-traceId', data: 'trace-1' },
    { type: 'finish' },
  ];
  await page.route('**/api/chat', (route) =>
    route.fulfill({
      headers: {
        'content-type': 'text/event-stream',
        'x-vercel-ai-ui-message-stream': 'v1',
      },
      body: `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('')}data: [DONE]\n\n`,
    }),
  );

  const chatPage = new ChatPage(page);
  await chatPage.createNewChat();
  await chatPage.sendUserMessage('Hello');
  await chatPage.isGenerationComplete();
  await expect(page.getByText('Streamed response')).toBeVisible();
  await expect(page.getByTestId('thumbs-up-button')).toHaveCount(0);
  await expect(page.getByTestId('thumbs-down-button')).toHaveCount(0);
  expect(feedbackRequests).toEqual([]);
});

for (const chatHistory of [false, true, undefined]) {
  test(`feedback controls with chat history ${chatHistory ?? 'unknown'}`, async ({
    adaContext,
  }) => {
    const { page } = adaContext;
    const feedbackRequests: { method: string; body?: unknown }[] = [];

    await page.route('**/api/config', async (route) => {
      if (chatHistory === undefined) {
        await route.fulfill({ status: 500, json: { code: 'offline:chat' } });
      } else {
        await route.fulfill({
          json: { features: { chatHistory, feedback: true } },
        });
      }
    });
    await page.route(`**/api/chat/${chatId}`, (route) =>
      route.fulfill({
        json: { id: chatId, title: 'Test chat', visibility: 'private' },
      }),
    );
    await page.route(`**/api/chat/${chatId}/stream`, (route) =>
      route.fulfill({ status: 204 }),
    );
    await page.route(`**/api/messages/${chatId}`, (route) =>
      route.fulfill({
        json: [
          {
            id: messageId,
            role: 'assistant',
            parts: [{ type: 'text', text: 'Assistant response' }],
            createdAt: '2025-01-01T00:00:00.000Z',
          },
        ],
      }),
    );
    await page.route('**/api/feedback**', async (route) => {
      const request = route.request();
      feedbackRequests.push({
        method: request.method(),
        body: request.method() === 'POST' ? request.postDataJSON() : undefined,
      });
      await route.fulfill({ json: {} });
    });

    await page.goto(`/chat/${chatId}`);
    await expect(page.getByText('Assistant response')).toBeVisible();
    const up = page.getByTestId('thumbs-up-button');
    const down = page.getByTestId('thumbs-down-button');

    if (chatHistory === true) {
      await expect(up).toBeVisible();
      await expect(down).toBeVisible();
      await up.click();
      await expect(up).toHaveClass(/text-green-600/);
      await down.click();
      await expect(down).toHaveClass(/text-red-600/);
      expect(feedbackRequests).toEqual([
        { method: 'GET', body: undefined },
        { method: 'POST', body: { messageId, feedbackType: 'thumbs_up' } },
        { method: 'POST', body: { messageId, feedbackType: 'thumbs_down' } },
      ]);
    } else {
      await expect(up).toHaveCount(0);
      await expect(down).toHaveCount(0);
      expect(feedbackRequests).toEqual([]);
    }
  });
}
