async function renderExecutionDiagrams() {
    const blocks = document.querySelectorAll('pre code.language-mermaid');
    if (!blocks.length) return;
    try {
        const { default: mermaid } = await import(
            'https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs'
        );
        mermaid.initialize({ startOnLoad: false, securityLevel: 'strict' });
        const nodes = Array.from(blocks, (block) => {
            const diagram = document.createElement('div');
            diagram.className = 'mermaid';
            diagram.style.overflowX = 'auto';
            diagram.textContent = block.textContent;
            block.parentElement.replaceWith(diagram);
            return diagram;
        });
        await mermaid.run({ nodes });
    } catch (error) {
        console.error('Could not render Mermaid diagrams:', error);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderExecutionDiagrams);
} else {
    renderExecutionDiagrams();
}
