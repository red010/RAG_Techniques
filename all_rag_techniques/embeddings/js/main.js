/**
 * 3D Recipe Embeddings Visualization - Main Three.js Logic
 * 
 * Features:
 * - Interactive 3D scene with orbit controls
 * - Hover tooltips showing recipe names
 * - Real-time search filtering
 * - K-nearest neighbors visualization on click
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ============================================================================
// GLOBAL STATE
// ============================================================================

let scene, camera, renderer, controls;
let raycaster, mouse, tooltip, searchInput;
let recipePoints = []; // Array of {mesh, name, position, category}
let hoveredObject = null;
let selectedObject = null;
let nearestNeighbors = [];

// Category color scheme - will be populated dynamically
let CATEGORY_COLORS = {};

// Default color palette for automatic category coloring
const COLOR_PALETTE = [
    0xFFD700,   // Gold
    0xFF6347,   // Tomato
    0x4169E1,   // Royal Blue
    0x32CD32,   // Lime Green
    0xFF8C00,   // Dark Orange
    0x9370DB,   // Medium Purple
    0xFF69B4,   // Hot Pink
    0x00CED1,   // Dark Turquoise
    0xFF1493,   // Deep Pink
    0x7FFF00,   // Chartreuse
    0xFF4500,   // Orange Red
    0xDA70D6,   // Orchid
    0x20B2AA,   // Light Sea Green
    0xFFD700,   // Gold
    0xF08080,   // Light Coral
];

// Interaction colors
const INTERACTION_COLORS = {
    hover: 0xfeca57,        // Yellow
    selected: 0xff0000,     // Red (changed from Magenta)
    neighbor: 0x00ff00,     // Green (changed from Cyan)
    searchMatch: 0x4cd137,  // Green
    dimmed: 0x666666        // Gray
};

// ============================================================================
// INITIALIZATION
// ============================================================================

async function init() {
    // Setup Three.js scene
    setupScene();
    setupCamera();
    setupRenderer();
    setupLights();
    setupControls();
    
    // Setup interactivity
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    tooltip = document.getElementById('tooltip');
    searchInput = document.getElementById('searchInput');
    
    // Event listeners
    window.addEventListener('resize', onWindowResize);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('click', onClick);
    searchInput.addEventListener('input', onSearch);
    
    // Load and render data
    await loadRecipeData();
    
    // Hide loading indicator
    document.getElementById('loading').style.display = 'none';
    
    // Start animation loop
    animate();
}

function setupScene() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
}

function setupCamera() {
    camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera.position.z = 15;
    camera.position.y = 5;
}

function setupRenderer() {
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById('canvas-container').appendChild(renderer.domElement);
}

function setupLights() {
    // Ambient light for overall illumination
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    // Point light for highlights
    const pointLight = new THREE.PointLight(0xffffff, 0.8);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);
    
    // Additional fill light
    const fillLight = new THREE.PointLight(0xff6b35, 0.4);
    fillLight.position.set(-10, -10, -10);
    scene.add(fillLight);
}

function setupControls() {
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 5;
    controls.maxDistance = 50;
}

// ============================================================================
// CATEGORY COLOR MANAGEMENT
// ============================================================================

/**
 * Extract unique categories from dataset and assign colors
 */
function extractCategoriesAndColors(items) {
    const uniqueCategories = [...new Set(items.map(item => item.category))];
    
    CATEGORY_COLORS = {};
    uniqueCategories.forEach((category, index) => {
        CATEGORY_COLORS[category] = COLOR_PALETTE[index % COLOR_PALETTE.length];
    });
    
    console.log(`✓ Detected ${uniqueCategories.length} categories:`, uniqueCategories);
    console.log('Color mapping:', CATEGORY_COLORS);
    
    return uniqueCategories;
}

/**
 * Get formatted label for a category
 */
function getCategoryLabel(category) {
    const categoryLabels = {
        // Italian recipes
        primi: 'Primi Piatti',
        secondi_carne: 'Secondi Carne',
        secondi_pesce: 'Secondi Pesce',
        contorni: 'Contorni',
        pizza: 'Pizza/Focaccia',
        antipasti: 'Antipasti',
        dolci: 'Dolci',
        // Movies
        action: 'Action',
        comedy: 'Comedy',
        drama: 'Drama',
        scifi: 'Sci-Fi',
        horror: 'Horror',
        romance: 'Romance',
        thriller: 'Thriller',
        animation: 'Animation',
        mystery: 'Mystery',
        fantasy: 'Fantasy',
        crime: 'Crime',
        adventure: 'Adventure',
        war: 'War',
        western: 'Western',
        musical: 'Musical',
        biography: 'Biography',
    };
    
    // Return mapped label or capitalize the category
    return categoryLabels[category] || 
           category.charAt(0).toUpperCase() + category.slice(1).replace(/_/g, ' ');
}

/**
 * Update the legend panel with current categories
 */
function updateLegend(categories) {
    const legendItems = document.getElementById('legendItems');
    if (!legendItems) return;
    
    // Clear existing items
    legendItems.innerHTML = '';
    
    // Sort categories alphabetically
    const sortedCategories = [...categories].sort();
    
    // Create legend items
    sortedCategories.forEach(category => {
        const color = CATEGORY_COLORS[category];
        const hexColor = '#' + color.toString(16).padStart(6, '0');
        const label = getCategoryLabel(category);
        
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `
            <span class="legend-color" style="background: ${hexColor};"></span>
            <span>${label}</span>
        `;
        legendItems.appendChild(item);
    });
    
    console.log('✓ Legend updated with', categories.length, 'categories');
}

/**
 * Update the page title with dataset description
 */
function updateTitle(datasetInfo) {
    const titleElement = document.getElementById('datasetTitle');
    if (!titleElement) return;
    
    if (datasetInfo && datasetInfo.description) {
        // Choose emoji based on language or keep generic
        const emoji = datasetInfo.language === 'it' ? '🍝' : '📊';
        titleElement.textContent = `${emoji} ${datasetInfo.description}`;
    } else {
        titleElement.textContent = '📊 Dataset Embeddings Explorer';
    }
    
    console.log('✓ Title updated');
}

// ============================================================================
// DATA LOADING
// ============================================================================

async function loadRecipeData() {
    try {
        console.log('🔄 Loading data from ./public/data.json...');
        const response = await fetch('./public/data.json');
        console.log('✓ Fetch response received:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const dataset = await response.json();
        console.log('✓ JSON parsed. Type:', Array.isArray(dataset) ? 'array' : 'object');
        console.log('Dataset structure:', {
            isArray: Array.isArray(dataset),
            hasItems: !!dataset.items,
            hasDatasetInfo: !!dataset.dataset_info
        });
        
        // Support both old format (array) and new format (object with items)
        const items = Array.isArray(dataset) ? dataset : dataset.items;
        
        if (!items || items.length === 0) {
            console.error('❌ No items found in dataset');
            throw new Error('No items found in dataset');
        }
        
        console.log(`✓ Loaded ${items.length} items`);
        if (dataset.dataset_info) {
            console.log(`📊 Dataset: ${dataset.dataset_info.description}`);
        }
        
        // Debug first item
        console.log('First item structure:', items[0]);
        
        // Extract categories and assign colors dynamically
        const categories = extractCategoriesAndColors(items);
        
        // Update UI with dataset info
        updateTitle(dataset.dataset_info);
        updateLegend(categories);
        
        // Create 3D points for each item
        items.forEach((item, index) => {
            // Support both formats: name as string (old) or name.local (new)
            const name = typeof item.name === 'string' ? item.name : item.name.local;
            if (index === 0) {
                console.log(`Creating point for: ${name}`);
            }
            createRecipePoint(name, item.position, item.category, item);
        });
        
        console.log(`✓ Created ${recipePoints.length} 3D points`);
        
        // Center the camera on the data
        centerCameraOnPoints();
        console.log('✓ Camera centered on data');
        
    } catch (error) {
        console.error('❌ Error loading recipe data:', error);
        console.error('Error details:', {
            message: error.message,
            stack: error.stack
        });
        alert('Failed to load recipe data. Check console for details.');
    }
}

function createRecipePoint(name, position, category, fullItemData) {
    // Get color based on category
    const categoryColor = CATEGORY_COLORS[category] || 0xff6b35;
    
    // Create sphere geometry (smaller size)
    const geometry = new THREE.SphereGeometry(0.08, 32, 32);
    const material = new THREE.MeshStandardMaterial({
        color: categoryColor,
        emissive: categoryColor,
        emissiveIntensity: 0.3,
        metalness: 0.5,
        roughness: 0.5
    });
    
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(position[0], position[1], position[2]);
    
    // Store full metadata for tooltip
    mesh.userData = { 
        name, 
        category,
        originalPosition: position,
        originalColor: categoryColor,
        fullData: fullItemData  // Store complete item data
    };
    
    scene.add(mesh);
    recipePoints.push({
        mesh,
        name,
        category,
        position: new THREE.Vector3(position[0], position[1], position[2])
    });
}

function centerCameraOnPoints() {
    if (recipePoints.length === 0) return;
    
    // Calculate bounding box
    const box = new THREE.Box3();
    recipePoints.forEach(point => box.expandByPoint(point.position));
    
    const center = new THREE.Vector3();
    box.getCenter(center);
    
    const size = new THREE.Vector3();
    box.getSize(size);
    
    // Position camera to see all points
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= 1.5; // Add some padding
    
    camera.position.set(center.x, center.y, center.z + cameraZ);
    controls.target.copy(center);
    controls.update();
}

// ============================================================================
// INTERACTIVITY
// ============================================================================

function buildTooltipHTML(userData) {
    /**
     * Build rich HTML tooltip with all item properties
     */
    const data = userData.fullData;
    
    // Start with title
    let html = `<div class="tooltip-title">${userData.name}</div>`;
    
    if (!data) {
        return html;
    }
    
    // Add English name if available
    if (data.name && data.name.en && data.name.en !== userData.name) {
        html += `<div class="tooltip-field">
            <span class="tooltip-label">English:</span> 
            <span class="tooltip-value">${data.name.en}</span>
        </div>`;
    }
    
    // Add category
    if (data.category) {
        const categoryLabel = getCategoryLabel(data.category);
        
        html += `<div class="tooltip-field">
            <span class="tooltip-label">Category:</span> 
            <span class="tooltip-value">${categoryLabel}</span>
        </div>`;
    }
    
    // Add description if available
    if (data.description) {
        const shortDesc = data.description.length > 100 
            ? data.description.substring(0, 100) + '...' 
            : data.description;
        html += `<div class="tooltip-field">
            <span class="tooltip-label">Description:</span> 
            <span class="tooltip-value">${shortDesc}</span>
        </div>`;
    }
    
    // Add metadata fields (region, difficulty, etc.)
    if (data.metadata) {
        for (const [key, value] of Object.entries(data.metadata)) {
            const labelKey = key.charAt(0).toUpperCase() + key.slice(1);
            html += `<div class="tooltip-field">
                <span class="tooltip-label">${labelKey}:</span> 
                <span class="tooltip-value">${value}</span>
            </div>`;
        }
    }
    
    return html;
}

function onMouseMove(event) {
    // Update mouse coordinates for raycasting
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    
    // Update raycaster
    raycaster.setFromCamera(mouse, camera);
    
    // Check for intersections
    const meshes = recipePoints.map(p => p.mesh);
    const intersects = raycaster.intersectObjects(meshes);
    
    if (intersects.length > 0) {
        const intersectedObject = intersects[0].object;
        
        // Show rich tooltip with all properties
        tooltip.innerHTML = buildTooltipHTML(intersectedObject.userData);
        tooltip.style.left = event.clientX + 15 + 'px';
        tooltip.style.top = event.clientY + 15 + 'px';
        tooltip.classList.add('visible');
        
        // Only handle hover effects if there's NO active selection
        if (!selectedObject) {
            // Highlight hovered object
            if (hoveredObject !== intersectedObject) {
                if (hoveredObject) {
                    resetPointColor(hoveredObject);
                }
                hoveredObject = intersectedObject;
                setPointColor(hoveredObject, INTERACTION_COLORS.hover, 0.8);
            }
        }
        
        // Change cursor
        renderer.domElement.style.cursor = 'pointer';
    } else {
        // Hide tooltip
        tooltip.classList.remove('visible');
        
        // Only reset hover if there's NO active selection
        if (!selectedObject) {
            if (hoveredObject) {
                resetPointColor(hoveredObject);
                hoveredObject = null;
            }
        }
        
        // Reset cursor
        renderer.domElement.style.cursor = 'grab';
    }
}

function onClick(event) {
    // Update raycaster
    raycaster.setFromCamera(mouse, camera);
    
    const meshes = recipePoints.map(p => p.mesh);
    const intersects = raycaster.intersectObjects(meshes);
    
    if (intersects.length > 0) {
        // Clicked on a recipe point
        const clickedObject = intersects[0].object;
        selectPointAndNeighbors(clickedObject);
    } else {
        // Clicked on background - reset selection
        resetSelection();
    }
}

function onSearch(event) {
    const query = event.target.value.toLowerCase().trim();
    
    if (query === '') {
        // Don't reset if user is not actively clearing the search
        // (this prevents interference with click selection)
        // Only reset to category colors if there's no active selection
        if (!selectedObject) {
            recipePoints.forEach(point => {
                resetPointColor(point.mesh);
            });
        }
        return;
    }
    
    // Clear previous selection
    selectedObject = null;
    nearestNeighbors = [];
    hoveredObject = null;
    
    // Filter and highlight matching recipes
    recipePoints.forEach(point => {
        const matches = point.name.toLowerCase().includes(query);
        
        if (matches) {
            // Highlight matching recipes
            setPointColor(point.mesh, INTERACTION_COLORS.searchMatch, 0.9);
        } else {
            // Dim non-matching recipes
            setPointColor(point.mesh, INTERACTION_COLORS.dimmed, 0.3);
        }
    });
}

// ============================================================================
// K-NEAREST NEIGHBORS
// ============================================================================

function selectPointAndNeighbors(clickedMesh) {
    // Reset previous selection
    resetSelection();
    
    // Set as selected
    selectedObject = clickedMesh;
    setPointColor(selectedObject, INTERACTION_COLORS.selected, 1.0);
    
    // Find clicked point data
    const clickedPoint = recipePoints.find(p => p.mesh === clickedMesh);
    if (!clickedPoint) return;
    
    // Calculate distances to all other points
    const distances = recipePoints
        .filter(p => p.mesh !== clickedMesh)
        .map(p => ({
            point: p,
            distance: clickedPoint.position.distanceTo(p.position)
        }))
        .sort((a, b) => a.distance - b.distance);
    
    // Select 5 nearest neighbors
    nearestNeighbors = distances.slice(0, 5).map(d => d.point.mesh);
    
    // Highlight neighbors
    nearestNeighbors.forEach(mesh => {
        setPointColor(mesh, INTERACTION_COLORS.neighbor, 0.9);
    });
    
    // Dim all other points
    recipePoints.forEach(point => {
        if (point.mesh !== selectedObject && !nearestNeighbors.includes(point.mesh)) {
            setPointColor(point.mesh, INTERACTION_COLORS.dimmed, 0.2);
        }
    });
    
    // Log to console
    console.log(`Selected: ${clickedPoint.name} (${clickedPoint.category})`);
    console.log('5 Nearest neighbors:');
    distances.slice(0, 5).forEach((d, i) => {
        console.log(`  ${i + 1}. ${d.point.name} (${d.point.category}) - distance: ${d.distance.toFixed(2)}`);
    });
}

function resetSelection() {
    selectedObject = null;
    nearestNeighbors = [];
    hoveredObject = null;  // Also reset hover
    searchInput.value = '';
    
    // Reset all points to default color
    recipePoints.forEach(point => {
        resetPointColor(point.mesh);
    });
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function setPointColor(mesh, color, emissiveIntensity) {
    mesh.material.color.setHex(color);
    mesh.material.emissive.setHex(color);
    mesh.material.emissiveIntensity = emissiveIntensity;
    mesh.material.opacity = 1.0;
    mesh.material.transparent = false;
}

function resetPointColor(mesh) {
    // Restore original category color
    const originalColor = mesh.userData.originalColor || 0xff6b35;
    mesh.material.color.setHex(originalColor);
    mesh.material.emissive.setHex(originalColor);
    mesh.material.emissiveIntensity = 0.3;
    mesh.material.opacity = 1.0;
    mesh.material.transparent = false;
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// ============================================================================
// ANIMATION LOOP
// ============================================================================

function animate() {
    requestAnimationFrame(animate);
    
    // Update controls
    controls.update();
    
    // Render scene
    renderer.render(scene, camera);
}

// ============================================================================
// START APPLICATION
// ============================================================================

init();

