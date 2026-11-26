<script lang="ts">
	import { tr, t } from '$lib/stores/locale.svelte';
	import { serverStore } from '$lib/stores';

	interface Props {
		isEmpty: boolean;
	}

	let { isEmpty = false }: Props = $props();
</script>

<div
	class={[
		'pointer-events-none mb-4 hidden px-4 text-center text-balance',
		isEmpty && 'mb-[calc(50dvh-8rem)] md:mb-8 pointer-events-auto block!'
	]}
>
	<h1 class="mb-2 text-2xl font-semibold tracking-tight md:text-3xl">{tr("Hello there")}</h1>

	<p class="text-muted-foreground md:text-lg">
		{serverStore.props?.modalities?.vision
			? t('输入消息，或添加图片', 'Enter a message or attach an image')
			: t('输入消息，开始对话', 'Enter a message to start chatting')}
	</p>
</div>
