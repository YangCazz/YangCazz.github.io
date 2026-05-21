// 侧边栏导航系统
function updateSidebarActive() {
    const currentPath = window.location.pathname;
    const currentHash = window.location.hash;
    const sidebarLinks = document.querySelectorAll('.sidebar-nav .nav-link');

    sidebarLinks.forEach(link => {
        link.classList.remove('active');
    });

    const hasPageNavigation = document.querySelectorAll('.sidebar-nav .nav-link[href^="#"]').length > 0;

    if (hasPageNavigation) {
        updateNavOnScroll();
    } else {
        sidebarLinks.forEach(link => {
            if (link.href && link.href.includes('http')) {
                const linkPath = new URL(link.href).pathname;
                if (linkPath === currentPath || (currentPath === '/' && linkPath === '/')) {
                    link.classList.add('active');
                }
            }
        });
    }
}

function updateNavOnScroll() {
    const sections = document.querySelectorAll('section[id]');
    let currentSection = '';
    let minDistance = Infinity;

    const isAtTop = window.scrollY < 100;

    if (isAtTop && sections.length > 0) {
        currentSection = '#' + sections[0].getAttribute('id');
    } else {
        sections.forEach(section => {
            const sectionTop = section.getBoundingClientRect().top;
            const offset = 100;
            if (sectionTop < offset && Math.abs(sectionTop) < minDistance) {
                minDistance = Math.abs(sectionTop);
                currentSection = '#' + section.getAttribute('id');
            }
        });
    }

    if (currentSection) {
        const sidebarLinks = document.querySelectorAll('.sidebar-nav .nav-link[href^="#"]');
        sidebarLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === currentSection) {
                link.classList.add('active');
            }
        });
    }
}

function initializeNavigation() {
    const sidebarLinks = document.querySelectorAll('.sidebar-nav .nav-link');
    sidebarLinks.forEach(link => {
        link.classList.remove('active');
    });

    const hasPageNavigation = document.querySelectorAll('.sidebar-nav .nav-link[href^="#"]').length > 0;

    if (hasPageNavigation) {
        const firstSection = document.querySelector('section[id]');
        if (firstSection) {
            const firstLink = document.querySelector('.sidebar-nav .nav-link[href="#' + firstSection.getAttribute('id') + '"]');
            if (firstLink) {
                firstLink.classList.add('active');
            }
        }
    } else {
        updateSidebarActive();
    }
}

// 平滑滚动
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// 汉堡菜单切换
document.addEventListener('DOMContentLoaded', function() {
    var toggle = document.querySelector('.nav-toggle');
    var menu = document.querySelector('.nav-menu');
    if (toggle && menu) {
        toggle.addEventListener('click', function() {
            menu.classList.toggle('open');
            toggle.classList.toggle('open');
        });
        // 点击菜单外关闭
        document.addEventListener('click', function(e) {
            if (!toggle.contains(e.target) && !menu.contains(e.target)) {
                menu.classList.remove('open');
                toggle.classList.remove('open');
            }
        });
    }
});

// 事件监听
window.addEventListener('scroll', updateNavOnScroll);
window.addEventListener('hashchange', updateSidebarActive);
window.addEventListener('popstate', initializeNavigation);

document.addEventListener('visibilitychange', function() {
    if (!document.hidden) {
        setTimeout(initializeNavigation, 100);
    }
});
